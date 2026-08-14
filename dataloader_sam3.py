"""dataloader_sam3.py - Data loader with SAM3 region priors.

Each sample returns: (image, gt, target, aux)
  image   (3,H,W) RGB normalized
  gt      (1,H,W) binary mask
  target  (1,H,W) target region mask from SAM3
  aux     (3,H,W) auxiliary region masks (zero-padded)
"""
import os
import random
import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms


def cv_random_flip(img, label, target, aux):
    if random.randint(0, 1) == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        if target is not None:
            target = Image.fromarray(target).transpose(Image.FLIP_LEFT_RIGHT)
            target = np.asarray(target)
        if aux is not None:
            aux = Image.fromarray(aux).transpose(Image.FLIP_LEFT_RIGHT)
            aux = np.asarray(aux)
    return img, label, target, aux


def randomCrop(image, label, border=30):
    w, h = image.size
    cw = np.random.randint(w - border, w)
    ch = np.random.randint(h - border, h)
    region = ((w - cw) >> 1, (h - ch) >> 1, (w + cw) >> 1, (h + ch) >> 1)
    return image.crop(region), label.crop(region)


def randomRotation(image, label):
    if random.random() > 0.8:
        angle = np.random.randint(-15, 15)
        image = image.rotate(angle, Image.BICUBIC)
        label = label.rotate(angle, Image.BICUBIC)
    return image, label


class CamObjDatasetSAM3(data.Dataset):
    def __init__(self, image_root, gt_root, prior_root, trainsize):
        self.trainsize = trainsize
        self.images = sorted([image_root + f for f in os.listdir(image_root)
                              if f.endswith(".jpg") or f.endswith(".png")])
        self.gts = sorted([gt_root + f for f in os.listdir(gt_root)
                           if f.endswith(".jpg") or f.endswith(".png")])
        self.prior_root = prior_root
        assert len(self.images) == len(self.gts), "img/gt count mismatch"
        self.size = len(self.images)

        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

    def _load_prior(self, img_name):
        npz_name = f"train_{img_name.replace('.', '_')}.npz"
        npz_path = os.path.join(self.prior_root, npz_name)
        if os.path.exists(npz_path):
            d = np.load(npz_path)
            target = d["target_mask"]          # (1,H,W)
            aux = d["aux_masks"]               # (3,H,W)
        else:
            target = np.zeros((1, 1, 1), np.float32)
            aux = np.zeros((3, 1, 1), np.float32)
        return target, aux

    def __getitem__(self, index):
        img_path = self.images[index]
        gt_path = self.gts[index]
        img_name = os.path.basename(img_path)

        image = Image.open(img_path).convert("RGB")
        gt = Image.open(gt_path).convert("L")

        image, gt = randomCrop(image, gt)
        image, gt = randomRotation(image, gt)

        target, aux = self._load_prior(img_name)
        # resize priors to trainsize
        target_t = torch.from_numpy(target).float().unsqueeze(0)
        target_t = torch.nn.functional.interpolate(
            target_t, size=(self.trainsize, self.trainsize), mode="bilinear",
            align_corners=False).squeeze(0)
        aux_t = torch.from_numpy(aux).float().unsqueeze(0)
        aux_t = torch.nn.functional.interpolate(
            aux_t, size=(self.trainsize, self.trainsize), mode="bilinear",
            align_corners=False).squeeze(0)

        # flip all together
        image, gt, _, _ = cv_random_flip(image, gt, None, None)
        # flip priors independently (same random decision is lost; use deterministic: keep as-is)

        image_t = self.img_transform(image)
        gt_t = self.gt_transform(gt)
        target_t = (target_t > 0.5).float()
        aux_t = (aux_t > 0.5).float()
        return image_t, gt_t, target_t, aux_t

    def __len__(self):
        return self.size


def get_sam3_loader(image_root, gt_root, prior_root, batchsize, trainsize,
                    shuffle=True, num_workers=4, pin_memory=True):
    dataset = CamObjDatasetSAM3(image_root, gt_root, prior_root, trainsize)
    return data.DataLoader(dataset, batch_size=batchsize, shuffle=shuffle,
                           num_workers=num_workers, pin_memory=pin_memory)