"""dataloader_sam3.py - Data loader with SAM3 region priors (CAMO + COD10K).

Each sample returns: (image, gt, target, aux)
  image   (3,H,W) RGB normalized
  gt      (1,H,W) binary mask
  target  (1,H,W) target region mask from SAM3
  aux     (3,H,W) auxiliary region masks (zero-padded)

Improvements over the original loader:
  * image <-> GT are paired by basename (not by sorted index) -> no silent
    misalignment when the dataset is merged (e.g. CAMO + COD10K).
  * randomCrop / randomRotation / randomFlip are applied to image, GT and the
    prior masks together, so target/aux stay aligned with the augmented image.
  * `split` controls the npz prefix (train_ / test_).
"""
import os
import random
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms

from utils import seed_worker


def cv_random_flip(img, label, target, aux):
    if random.randint(0, 1) == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        target = target.transpose(Image.FLIP_LEFT_RIGHT)
        aux = aux.transpose(Image.FLIP_LEFT_RIGHT)
    return img, label, target, aux


def randomCrop(image, label, target, aux, border=30):
    w, h = image.size
    if w <= border or h <= border:          # tiny image: keep as-is
        return image, label, target, aux
    cw = np.random.randint(w - border, w)
    ch = np.random.randint(h - border, h)
    region = ((w - cw) >> 1, (h - ch) >> 1, (w + cw) >> 1, (h + ch) >> 1)
    return (image.crop(region), label.crop(region),
            target.crop(region), aux.crop(region))


def randomRotation(image, label, target, aux):
    if random.random() > 0.8:
        angle = np.random.randint(-15, 15)
        image = image.rotate(angle, Image.BICUBIC)
        label = label.rotate(angle, Image.BICUBIC)
        target = target.rotate(angle, Image.BILINEAR)
        aux = aux.rotate(angle, Image.BILINEAR)
    return image, label, target, aux


def _prior_to_pil(mask, n_ch):
    """Convert a (C,H,W) float mask in [0,1] to a PIL image so geometric
    augmentations act on all channels identically to the RGB image."""
    m = np.clip(mask, 0.0, 1.0) * 255.0
    m = np.uint8(np.rint(m))
    if n_ch == 1:
        return Image.fromarray(m[0])
    return Image.fromarray(m.transpose(1, 2, 0))   # (H,W,3) -> RGB


def _pil_to_prior(pil_img, n_ch):
    """Inverse of _prior_to_pil: back to (C,H,W) float32 [0,1] tensor."""
    arr = np.asarray(pil_img, dtype=np.float32) / 255.0
    if n_ch == 1:
        arr = arr[None]
    else:
        arr = arr.transpose(2, 0, 1)
    return torch.from_numpy(arr).float()



def is_noncam(name):
    """True for COD10K 'NonCAM' images (neg_flag = 1 for the 8th input channel).

    SINGLE SOURCE OF TRUTH: the training loader and every inference path
    (MyTest_SAM3 / MyTrain_SAM3 / web_server_sam3) must use this rule, else the
    network is fed a different 8th channel at test time than during training.

    NOTE (label leakage, 2026-10): the class is encoded in the *file name*
    ("..._NonCAM_..."), so "label" mode hands the answer to the network.  It is
    kept as the default only to keep old checkpoints usable; every ablation /
    text arm must run with ``--neg_flag zero``.
    """
    return "NonCAM" in os.path.basename(name)


def neg_flag_value(name, mode="label"):
    """8th-channel scalar for one image.

    mode
      label - (historic default) 1.0 for a COD10K 'NonCAM' image, else 0.0.
              Reads the answer out of the file name -> optimistic on COD10K.
      zero  - always 0.0 (no leak; required for the 2276-image CAMO+COD10K-CAM
              table and for every prior / text ablation)
    """
    if mode == "zero":
        return 0.0
    if mode == "label":
        return 1.0 if is_noncam(name) else 0.0
    raise ValueError("neg_flag mode must be 'label' or 'zero', got %r" % (mode,))


def resize_prior_masks(target, aux, hw, mode="nearest", thresh=0.5):
    """Resize (1,H,W)/(3,H,W) prior masks to `hw` in ONE step -> binary float32.

    Two bugs this replaces:

    * the PIL round-trip resized the prior to the *native* image size first, so a
      prior stored at 264/448 px (``tools/compact_priors.py``) was up-sampled by
      up to 13x and then down-sampled again by the augmentation.  With NEAREST
      the second resample reads the first one's blocks, so the mask gains / loses
      whole 13-px steps -- region edges come out stair-cased ("grey") and thin
      targets disappear.  Resizing npz -> working size once keeps the shape.
    * ``_prior_to_pil`` quantises to uint8, so a mask stored as 0/255 (rather
      than 0/1) tripped ``clip(m, 0, 1) * 255`` and became an all-foreground
      prior.  Values are normalised here, whatever the producer wrote.
    """
    h, w = int(hw[0]), int(hw[1])

    def _one(m):
        m = np.asarray(m, np.float32)
        if m.ndim == 2:
            m = m[None]
        if float(m.max()) > 1.5:          # stored on a 0..255 scale
            m = m / 255.0
        m = np.clip(m, 0.0, 1.0)
        if tuple(m.shape[-2:]) != (h, w):
            t = torch.from_numpy(m)[None]                     # (1,C,H,W)
            t = F.interpolate(t, size=(h, w), mode=mode)
            m = t[0].numpy()
        return (m > thresh).astype(np.float32)

    return _one(target), _one(aux)


def prepare_prior_at(target, aux, hw):
    """Resize prior masks to `hw` and binarize at 0.5 -> (target, aux).

    SINGLE SOURCE OF TRUTH for prior resizing.  Bilinear upsampling of a binary
    mask yields 0~1 grey boundary pixels: the training loader binarized after
    resizing while inference used to skip that step, so the prior distribution
    at test time differed from training.  ``resize_prior_masks`` additionally
    makes the *scale* of the stored npz irrelevant (see its docstring).
    """
    t = target.detach().cpu().numpy() if torch.is_tensor(target) else target
    a = aux.detach().cpu().numpy() if torch.is_tensor(aux) else aux
    # bilinear + >0.5 == the historical "resize then binarize" behaviour
    t, a = resize_prior_masks(t, a, hw, mode="bilinear")
    return torch.from_numpy(t), torch.from_numpy(a)


def _prior_npz_candidates(split, name, prior_root):
    """Candidate npz paths for `name`, most likely first.

    `name` is not necessarily a bare file name: ``dataloader.test_dataset``
    builds its entries by string concatenation and then takes
    ``path.split('/')[-1]``, so a root assembled with ``os.sep`` leaks the whole
    path into the name (and the loader rewrites '.jpg' -> '.png' as well).
    Both variants are resolved here; without this normalisation the prior lookup
    missed every file and the whole prior branch fell back to zeros.
    """
    stem, ext = os.path.splitext(os.path.basename(name.replace("\\", "/")))
    exts = []
    for e in (ext, ".jpg", ".png"):
        if e and e.lower() not in [x.lower() for x in exts]:
            exts.append(e)
    return [os.path.join(prior_root, f"{split}_{(stem + e).replace('.', '_')}.npz")
            for e in exts]


def load_test_prior(split, name, prior_root, hw):
    """Load the ``{split}_*.npz`` prior of one image -> (target, aux, found).

    ``found=False`` means the npz is missing and an all-zero prior is returned.
    Callers MUST surface that count: silently evaluating an all-zero prior is
    the Layer-0 failure this repo was built to prevent.
    """
    npz_path = next((p for p in _prior_npz_candidates(split, name, prior_root)
                     if os.path.exists(p)), None)
    if npz_path is not None:
        with np.load(npz_path, allow_pickle=True) as d:
            target = torch.from_numpy(d["target_mask"]).float()
            aux = torch.from_numpy(d["aux_masks"]).float()
        found = True
    else:
        target = torch.zeros(1, int(hw[0]), int(hw[1]))
        aux = torch.zeros(3, int(hw[0]), int(hw[1]))
        found = False
    target, aux = prepare_prior_at(target, aux, hw)
    return target, aux, found


class CamObjDatasetSAM3(data.Dataset):
    """prior_mode (ablation switch, default "real" = unchanged behaviour):
      real    - use prior_root/SAM3 masks (default)
      zero    - feed all-zero priors (isolates "does the prior branch help?")
      oracle  - feed the GT mask as the prior (upper bound of the prior branch)
      shuffle - feed another image's prior (negative control)
    img_filter: substring filter on image file name (e.g. "camourflage") to
                train/ablate on a subset only.
    aug_mode:
      fast   - (default) first down-scale image/GT/prior to `aug_size`
               (1.25 x trainsize, i.e. 448 for 352) and only then run
               crop/rotate/flip.  COD10K/CAMO images are up to 3592x2386, so
               the legacy path resized three mask planes of that size for
               every sample (measured p99 = 1019 ms, max = 1708 ms per
               sample); the augmented result is down-scaled to trainsize
               anyway, so the augmentation only needs to happen slightly
               above the final resolution.
      legacy - augment at native resolution (bit-exact with runs before
               2026-10-17; use it only to reproduce an old number).
    """

    def __init__(self, image_root, gt_root, prior_root, trainsize, split="train",
                 prior_mode="real", img_filter=None, seed=0, aug_mode="fast",
                 neg_flag_mode="label"):
        self.trainsize = trainsize
        assert aug_mode in ("fast", "legacy"), aug_mode
        self.aug_mode = aug_mode
        assert neg_flag_mode in ("label", "zero"), neg_flag_mode
        self.neg_flag_mode = neg_flag_mode
        # 352 -> 448 : keeps ~100 px of context for the random crop, rounded to
        # a multiple of 32 so the conv stack sees a friendly shape.
        self.aug_size = max(64, int(round(trainsize * 1.25 / 32.0)) * 32)
        self.split = split
        self.prior_root = prior_root
        assert prior_mode in ("real", "zero", "oracle", "shuffle"), prior_mode
        self.prior_mode = prior_mode
        self.img_filter = img_filter
        self.seed = seed
        self._rng = random.Random(seed)

        images = sorted(f for f in os.listdir(image_root)
                        if f.endswith(".jpg") or f.endswith(".png"))
        gts = sorted(f for f in os.listdir(gt_root)
                     if f.endswith(".jpg") or f.endswith(".png"))
        gt_by_name = {os.path.splitext(g)[0]: g for g in gts}

        # pair image <-> GT by basename; detect NONCAM for negative flag
        self.samples = []
        skipped = 0
        for img in images:
            stem = os.path.splitext(img)[0]
            if stem in gt_by_name:
                # shared rule with every inference path (see is_noncam above)
                noncam = is_noncam(img)
                self.samples.append((os.path.join(image_root, img),
                                     os.path.join(gt_root, gt_by_name[stem]),
                                     noncam))
            else:
                skipped += 1
        if skipped:
            print(f">>> WARNING: {skipped} images without matching GT skipped")
        if img_filter:
            n_all = len(self.samples)
            self.samples = [s for s in self.samples if img_filter in os.path.basename(s[0])]
            print(f">>> img_filter='{img_filter}': {len(self.samples)}/{n_all} samples kept")
        assert len(self.samples) > 0, "no valid image/GT pairs found"
        self.size = len(self.samples)
        # Layer-0: make the 8th channel visible (a silently dead neg_flag was a
        # real risk while the rule lived in two different copies of the code)
        n_noncam = sum(1 for s in self.samples if s[2])
        if neg_flag_mode == "zero":
            print(f">>> neg_flag: all-zero on all {self.size} images (--neg_flag zero; "
                  f"no file-name label leak; {n_noncam} 'NonCAM' images are still present)")
        else:
            print(f">>> neg_flag=1 on {n_noncam}/{self.size} images (COD10K 'NonCAM' rule)")
        if prior_mode != "real":
            print(f">>> prior_mode = '{prior_mode}' (ablation: prior_root is "
                  f"{'ignored' if prior_mode in ('zero', 'oracle') else 'shuffled'})")

        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

    def _load_prior(self, img_name):
        # `with` closes the mmap/file handle: the old bare np.load() leaked one
        # fd per sample and 8 workers x thousands of epochs would eventually
        # hit EMFILE.  Name resolution is shared with inference so both paths
        # find exactly the same npz.
        npz_path = next((p for p in _prior_npz_candidates(self.split, img_name,
                                                          self.prior_root)
                         if os.path.exists(p)), None)
        if npz_path is not None:
            with np.load(npz_path, allow_pickle=True) as d:
                target = d["target_mask"].astype(np.float32)   # (1,H,W)
                aux = d["aux_masks"].astype(np.float32)        # (3,H,W)
        else:
            target = np.zeros((1, 1, 1), np.float32)
            aux = np.zeros((3, 1, 1), np.float32)
        return target, aux

    def __getitem__(self, index):
        img_path, gt_path, is_noncam = self.samples[index]
        img_name = os.path.basename(img_path)

        image = Image.open(img_path).convert("RGB")
        gt = Image.open(gt_path).convert("L")

        # ---- priors: resize straight to the working resolution (one step) ----
        # The old path was npz -> PIL(uint8) -> native image size -> aug_size,
        # i.e. the same mask was resampled twice (see resize_prior_masks) and a
        # 448/264-px npz on a 3592-px image came out stair-cased.  Where the
        # prior comes from is unchanged; only the resampling is.
        target, aux = self._load_prior(img_name)          # (1,H,W) / (3,H,W)
        img_w, img_h = image.size
        work_w, work_h = ((self.aug_size, self.aug_size)
                          if self.aug_mode == "fast" else (img_w, img_h))
        if self.prior_mode == "zero":
            target = np.zeros((1, work_h, work_w), np.float32)
            aux = np.zeros((3, work_h, work_w), np.float32)
        elif self.prior_mode == "oracle":
            _g = (np.asarray(gt, np.float32) / 255.0 > 0.5).astype(np.float32)
            target, aux = resize_prior_masks(
                _g[None], np.zeros((3,) + _g.shape, np.float32), (work_h, work_w))
        elif self.prior_mode == "shuffle":
            _j = self._rng.randrange(len(self.samples))
            target, aux = self._load_prior(os.path.basename(self.samples[_j][0]))
        if self.prior_mode != "zero":
            target, aux = resize_prior_masks(target, aux, (work_h, work_w))
        target_pil = _prior_to_pil(target, 1)
        aux_pil = _prior_to_pil(aux, 3)
        if self.aug_mode == "fast":
            # Down-scale first, augment second: the augmentation chain ends with
            # transforms.Resize(trainsize), so working at ~1.25x trainsize loses
            # nothing but removes the 3592x2386 mask resizes entirely.
            image = image.resize((self.aug_size, self.aug_size), Image.BICUBIC)
            gt = gt.resize((self.aug_size, self.aug_size), Image.BILINEAR)

        image, gt, target_pil, aux_pil = randomCrop(image, gt, target_pil, aux_pil)
        image, gt, target_pil, aux_pil = randomRotation(image, gt, target_pil, aux_pil)
        image, gt, target_pil, aux_pil = cv_random_flip(image, gt, target_pil, aux_pil)

        image_t = self.img_transform(image)
        gt_t = self.gt_transform(gt)

        # resize priors to trainsize and binarize at 0.5 (shared code path --
        # inference uses the exact same prepare_prior_at, see the P0 fix)
        target_t = _pil_to_prior(target_pil, 1)
        aux_t = _pil_to_prior(aux_pil, 3)
        target_t, aux_t = prepare_prior_at(target_t, aux_t,
                                           (self.trainsize, self.trainsize))
        neg_flag = torch.full((1, self.trainsize, self.trainsize),
                              neg_flag_value(img_name, self.neg_flag_mode))
        return image_t, gt_t, target_t, aux_t, neg_flag

    def __len__(self):
        return self.size


def get_sam3_loader(image_root, gt_root, prior_root, batchsize, trainsize,
                    split="train", shuffle=True, num_workers=4, pin_memory=True,
                    prior_mode="real", img_filter=None, seed=0, aug_mode="fast",
                    mp_context=None, neg_flag_mode="label"):
    dataset = CamObjDatasetSAM3(image_root, gt_root, prior_root, trainsize,
                                split=split, prior_mode=prior_mode,
                                img_filter=img_filter, seed=seed,
                                aug_mode=aug_mode, neg_flag_mode=neg_flag_mode)
    # persistent_workers avoids re-forking + re-opening every npz handle each
    # epoch; measured [data] p99 dropped from 1019 ms to ~250 ms with 8 workers.
    kw = {}
    if num_workers > 0:
        kw["persistent_workers"] = True
        kw["prefetch_factor"] = 4
        if mp_context:
            kw["multiprocessing_context"] = mp_context
    # --seed must control the *batch order* too: without a seeded generator the
    # global RNG state at DataLoader construction decided it, so two runs with
    # the same --seed saw different batches (and `shuffle` mode different
    # priors).  worker_init_fn seeds numpy/random inside every worker.
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    return data.DataLoader(dataset, batch_size=batchsize, shuffle=shuffle,
                           num_workers=num_workers, pin_memory=pin_memory,
                           generator=gen, worker_init_fn=seed_worker, **kw)


def check_prior_coverage(prior_root, image_root, split="train", img_filter=None,
                         strict=False, exclude_noncam=True):
    """Fail-loud guard against a silently dead prior branch.

    Layer-0 evidence (2026-10): the historic "SAM3 prior" run was launched with
    ``--prior_root ./prior_sam3/`` - a directory that does not exist.  ``_load_prior``
    silently returns zeros for a missing file, so that model was trained AND
    evaluated with an all-zero prior for 100% of the samples; the reported "SAM3"
    number is therefore not a prior-ablation result at all.  Nothing in the old
    pipeline said a word about it.

    This helper counts, before training starts, how many images really have an
    npz in ``prior_root``.  With ``strict=True`` a 0% coverage raises instead of
    training a dead branch.

    Returns (n_have, n_total, first_missing_npz).
    """
    n_total = 0
    n_have = 0
    n_noncam = 0
    first_missing = None
    if not os.path.isdir(image_root):
        print(f">>> [prior-check] WARNING: image_root not found: {image_root}")
        return 0, 0, None
    for img in sorted(os.listdir(image_root)):
        if not img.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            continue
        if img_filter and img_filter not in img:
            continue
        # COD10K 'NonCAM' images are *by design* zero-prior + neg_flag=1 (Step1
        # skips them).  Counting them in the denominator made a healthy run
        # report 57.7% and train the eye to ignore this line.
        if exclude_noncam and is_noncam(img):
            n_noncam += 1
            continue
        n_total += 1
        # the loader resolves '{split}_{stem}_{ext}.npz' for both the real and
        # the rewritten extension; using one hard-coded name here reported a
        # healthy 100% coverage for data whose priors the loader could not find
        cand = _prior_npz_candidates(split, img, prior_root)
        hit = next((p for p in cand if os.path.exists(p)), None)
        if hit is not None:
            n_have += 1
        elif first_missing is None:
            first_missing = cand[0]
    ratio = (n_have / n_total) if n_total else 0.0
    print(f">>> [prior-check] prior_root='{prior_root}' "
          f"exists={os.path.isdir(prior_root)}")
    print(f">>> [prior-check] {split}: {n_have}/{n_total} annotatable images have "
          f"a prior npz ({100 * ratio:.1f}%)"
          + (f" | {n_noncam} NonCAM excluded (zero prior by design)"
             if n_noncam else ""))
    if first_missing:
        print(f">>> [prior-check] first missing file: {first_missing}")
    if n_total and ratio == 0.0:
        msg = (f"prior_root='{prior_root}' provides NO prior for split '{split}': the "
               f"entire prior branch will see zeros (equivalent to --prior_mode zero). "
               f"Run region_selector.py first, or pass --prior_mode zero if that is "
               f"what you intend.")
        if strict:
            raise FileNotFoundError(msg)
        print(">>> [prior-check] " + "!" * 66)
        print(">>> [prior-check] WARNING: " + msg)
        print(">>> [prior-check] " + "!" * 66)
    elif n_total and ratio < 0.5:
        print(f">>> [prior-check] WARNING: coverage is only {100 * ratio:.1f}%; most "
              f"samples fall back to an all-zero prior.")
    return n_have, n_total, first_missing
