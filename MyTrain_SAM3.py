"""MyTrain_SAM3.py - Train SAM3 region-prior guided network.

Usage:
  python MyTrain_SAM3.py --epoch 50 --batchsize 6 --prior_root ./prior_sam3/

Layer-0 fixes (2026-10):
  * priors now go through the single shared `dataloader_sam3.prepare_prior_at`
    (resize + binarize at 0.5) for training AND validation -- validation used to
    skip the binarization, so the MAE that selects KKK_best_*.pth was measured
    on a different prior distribution than training used.
  * validation neg_flag uses the shared COD10K 'NonCAM' rule (was hard-coded 0).
  * every epoch prints prior coverage + neg_flag positives, and the resumed
    checkpoint is channel-checked (a 7ch file can no longer load silently).

2026-10-17 throughput fixes:
  * DataParallel is now OPT-IN (`--dp`, default OFF).  Measured on the same
    box: single GPU 0.271 s/step (2 scales) vs DP over 2 GPUs 19.06 s/step --
    ~70x slower.  replicate() only reached 2.2 GB/s (PCIe gen1 x16, no NVLink)
    and ~2834 kernel launches per iteration fought over the GIL.
  * new flags: --dp, --size_rates "0.75,1", --amp (bf16 autocast).
  * hard guards: --batchsize >= 2 (CBAM BatchNorm2d sees [1,4,1,1] at bs=1)
    and, with --dp, batchsize % n_gpus == 0 with >= 2 samples per shard
    (torch.chunk silently drops the remainder: 6 over 4 GPUs leaves one idle).
  * the [Train] line now reports img/s next to s/step.

NOTE: results after these fixes are NOT directly comparable with older runs.
"""
import argparse
import math
import os
import shutil
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch import optim
from tqdm import tqdm

import logging
from dataloader_sam3 import (get_sam3_loader, check_prior_coverage,
                             load_test_prior, neg_flag_value)
from check_ckpt_channels import assert_state_dict_channels
from dataloader import test_dataset
from utils import (clip_gradient, AvgMeter, set_seed, write_run_manifest,
                   save_checkpoint, load_checkpoint, read_best, write_best,
                   find_latest_checkpoint, save_map, write_eval_manifest,
                   run_eval_metrics)
from Network_SAM3 import Network


def hybrid_e_loss(pred, mask, loss_mode="fixed"):
    """Hybrid E-loss with edge-aware weighting for sharper boundaries.

    loss_mode
      fixed  - (default since 2026-10) ``weit`` (structure 5x * Sobel 9x)
               really multiplies the per-pixel BCE.
      legacy - the pre-2026-10 code.  ``reduction="mean"`` returned a scalar, so
               ``(weit * wbce).sum((2,3)) / weit.sum((2,3))`` was algebraically
               the *unweighted* mean BCE: the Sobel/structure weights only ever
               reached the E-loss and wIoU terms.  Kept to reproduce old runs.
    """
    # ---- edge-aware weight: boundary pixels get higher loss weight ----
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=torch.float32, device=mask.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           dtype=torch.float32, device=mask.device).view(1, 1, 3, 3)
    gt_edge = torch.abs(F.conv2d(mask, sobel_x, padding=1)) + \
              torch.abs(F.conv2d(mask, sobel_y, padding=1))
    gt_edge = torch.clamp(gt_edge, 0, 1)
    edge_weight = 1 + 9 * gt_edge  # boundary=10, non-boundary=1

    # ---- original hybrid E-loss with combined weights ----
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    weit = weit * edge_weight  # combine: structure weight × edge weight

    if loss_mode == "legacy":
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction="mean")
        wbce = ((weit * wbce).sum(dim=(2, 3)) + 1e-8) / (weit.sum(dim=(2, 3)) + 1e-8)
    elif loss_mode == "fixed":
        bce = F.binary_cross_entropy_with_logits(pred, mask, reduction="none")
        wbce = (weit * bce).sum(dim=(2, 3)) / (weit.sum(dim=(2, 3)) + 1e-8)
    else:
        raise ValueError("loss_mode must be 'fixed' or 'legacy', got %r" % (loss_mode,))
    pred = torch.sigmoid(pred)
    # keepdim + expand instead of .repeat(): same values, no full-size copy
    mpred = pred.mean(dim=(2, 3), keepdim=True).expand_as(pred)
    phiFM = pred - mpred
    mmask = mask.mean(dim=(2, 3), keepdim=True).expand_as(mask)
    phiGT = mask - mmask
    EFM = (2.0 * phiFM * phiGT + 1e-8) / (phiFM * phiFM + phiGT * phiGT + 1e-8)
    QFM = (1 + EFM) * (1 + EFM) / 4.0
    eloss = 1.0 - QFM.mean(dim=(2, 3))
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1.0 - (inter + 1 + 1e-8) / (union - inter + 1 + 1e-8)
    return (wbce + eloss + wiou).mean()


def train(train_loader, model, optimizer, epoch, size_rates, total_step, opt):
    model.train()
    loss_all = 0
    epoch_step = 0
    loss_g_record, loss_c_record = AvgMeter(), AvgMeter()
    log_every = max(1, int(getattr(opt, "log_interval", 10)))
    use_amp = bool(getattr(opt, "amp", False))
    t_last = time.time()
    pbar = tqdm(enumerate(train_loader, start=1), total=len(train_loader),
                desc=f"Epoch {epoch:03d}/{opt.epoch:03d}", ncols=100,
                file=open(os.devnull, "w"))
    for i, (images, gts, targets, auxs, neg_flags) in pbar:
        # ---- backup originals so each scale starts from full-size ----
        images_bak, gts_bak = images, gts
        targets_bak, auxs_bak = targets, auxs
        neg_flags_bak = neg_flags

        for rate in size_rates:
            optimizer.zero_grad()
            # non_blocking: the loop already overlaps H2D with the previous
            # iteration's kernels, and pinned memory is on for the loader
            images = images_bak.cuda(non_blocking=True)
            gts = gts_bak.cuda(non_blocking=True)
            targets = targets_bak.cuda(non_blocking=True)
            auxs = auxs_bak.cuda(non_blocking=True)
            neg_flags = neg_flags_bak.cuda(non_blocking=True)

            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            images = F.interpolate(images, size=(trainsize, trainsize), mode="bicubic", align_corners=True)
            gts = F.interpolate(gts, size=(trainsize, trainsize), mode="bicubic", align_corners=True)
            targets = F.interpolate(targets, size=(trainsize, trainsize), mode="bilinear", align_corners=False)
            auxs = F.interpolate(auxs, size=(trainsize, trainsize), mode="bilinear", align_corners=False)
            neg_flags = F.interpolate(neg_flags, size=(trainsize, trainsize), mode="nearest")

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=use_amp):
                edge, c2, c3, c4 = model(images, targets, auxs, neg_flags)
            # Losses stay in fp32: hybrid_e_loss builds a Sobel / 31x31 avg-pool
            # weight and a per-image QFM ratio, both numerically unhappy in bf16.
            loss_g = hybrid_e_loss(edge.float(), gts, opt.loss_mode)
            loss_c2 = hybrid_e_loss(c2.float(), gts, opt.loss_mode)
            loss_c3 = hybrid_e_loss(c3.float(), gts, opt.loss_mode)
            loss_c4 = hybrid_e_loss(c4.float(), gts, opt.loss_mode)
            # 加权深监督：深层预测权重更高
            loss = 0.4 * loss_g + 0.25 * loss_c2 + 0.2 * loss_c3 + 0.15 * loss_c4
            loss_c = loss_c2 + loss_c3 + loss_c4

            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            epoch_step += 1
            loss_all += loss.data
            if rate == 1:
                loss_g_record.update(loss_g.data, opt.batchsize)
                loss_c_record.update(loss_c.data, opt.batchsize)

        if i % log_every == 0 or i == total_step:
            # s/step is the number that decides whether a run is worth starting:
            # 673 train batches x 0.4 s = ~4.5 min/epoch.  Keep it in log.log so
            # "is this run slow?" never needs a separate profiler again.
            now = time.time()
            dt = (now - t_last) / log_every
            t_last = now
            img_per_s = opt.batchsize / dt if dt > 0 else 0.0
            logging.info("[Train]:Epoch[{:03d}/{:03d}] Step[{:04d}/{:04d}] loss_g:{:.4f} loss_c:{:.4f} "
                         "| {:.3f}s/step {:.1f}img/s".format(
                             epoch, opt.epoch, i, total_step, loss_g_record.show(),
                             loss_c_record.show(), dt, img_per_s))

    loss_all /= epoch_step
    logging.info("[Train]:Epoch[{:03d}/{:03d}] Loss_AVG:{:.4f}".format(epoch, opt.epoch, loss_all))
    # NO checkpoint is written here: the periodic Net_epoch_*.pth needs best_state
    # (see the main loop).  A snapshot with best_mae=None was indistinguishable
    # from "this run never had a best epoch", so resuming from it either reset
    # best MAE or -- worse -- pointed final_eval at a KKK_best file for the wrong
    # epoch.


def _val_prior(prior_mode, name, gt_np, hw, prior_root, prev_name=None,
               split="test"):
    """Build (target, aux, found) for validation with the SAME switch as training.

    Layer-0 bug this replaces: ``test()`` always loaded the real npz, so a run
    started with ``--prior_mode zero`` trained a zero-prior network but selected
    ``KKK_best_*.pth`` on a *real-prior* MAE -- the headline number came from an
    input the model never saw.  ``shuffle`` had the same problem.

    ``gt_np`` is the [0,1] GT of the validation image (only "oracle" uses it).
    ``found`` is True for "zero"/"oracle" because those modes intentionally do
    not read an npz; reporting them as "missing" would trip --strict_prior.

    ``split`` selects the npz prefix (``test_*`` vs ``train_*``).  It used to be
    hard-coded to "test", so ``--val_split train`` silently scored every image
    with a *missing* (all-zero) prior: the split of the npz is part of the
    input, not an implementation detail.
    """
    hw = (int(hw[0]), int(hw[1]))
    if prior_mode == "zero":
        return torch.zeros(1, *hw), torch.zeros(3, *hw), True
    if prior_mode == "oracle":
        g = np.ascontiguousarray((gt_np > 0.5).astype(np.float32))
        g = torch.from_numpy(g).view(1, 1, *g.shape)
        g = F.interpolate(g, size=hw, mode="nearest").view(1, *hw)
        return g, torch.zeros(3, *hw), True
    if prior_mode == "shuffle":
        # deterministic negative control: the previous image's prior
        return load_test_prior(split, prev_name or name, prior_root, hw)
    return load_test_prior(split, name, prior_root, hw)


# (dataset, prior_mode) pairs for which the all-zero-prior warning was printed;
# the condition is a property of the data, not of the epoch, so it is reported
# once per run instead of on every validation pass.
_PRIOR_EMPTY_WARNED = set()


def test(val_loaders, model, epoch, save_path, best_state, prior_root,
         use_minmax=False, strict_prior=False, prior_mode="real",
         neg_flag_mode="label", split="test", save_pred_dir=None,
         select_best=True, opt=None):
    """Validate on one or more test datasets.

    val_loaders: list of (dataset_name, test_dataset) pairs.
    best model is selected by the mean MAE across all datasets.

    Layer-0 (2026-10): the prior fed here is built by ``_val_prior`` with the
    same ``prior_mode`` used for training.  Before this fix every run -- however
    it was trained -- was validated with the *real* prior, so model selection
    (and therefore "the" MAE of the run) was measured outside the run's own
    input distribution.  "zero"/"shuffle"/"oracle" arms are now self-consistent;
    a number produced by an old checkpoint must be regarded as a real-prior
    number until it is re-run.

    ``save_pred_dir`` writes one PNG per image (<dir>/<dataset>/<stem>.png) so
    ``eval_metrics.py`` can score the same forward pass with the 4-metric
    protocol instead of a second, subtly different inference script.
    ``select_best=False`` is used for that final pass: it must not overwrite
    KKK_best_*.pth.
    """
    model.eval()
    per_mae = {}
    with torch.no_grad():
        for dname, val_loader in val_loaders:
            mae_sum = 0
            n_prior_used, n_prior_missing, n_noncam_imgs = 0, 0, 0
            n_prior_empty = 0
            prev_name = None
            if hasattr(val_loader, "reset"):
                val_loader.reset()
            for i in range(val_loader.size):
                _, image, gt, name, _ = val_loader.load_data()
                gt_np = np.asarray(gt, np.float32) / 255.0  # normalize to [0,1]
                image = image.cuda(non_blocking=True)
                target, aux, _found = _val_prior(prior_mode, name, gt_np,
                                                 image.shape[-2:], prior_root,
                                                 prev_name=prev_name, split=split)
                prev_name = name
                n_prior_used += int(_found)
                n_prior_missing += int(not _found)
                # An existing npz can still carry an all-zero target mask: on
                # CAMO test 52/250 files in prior_sam3_0814 (79/250 in
                # prior_sam3_sel) are empty, i.e. the prior branch is dead for
                # those images even though the file is present.  Counting that
                # separately is the difference between "coverage" and "usable".
                if (prior_mode in ("real", "shuffle") and _found
                        and float(target.sum()) == 0.0):
                    n_prior_empty += 1
                target = target.unsqueeze(0).cuda(non_blocking=True)
                aux = aux.unsqueeze(0).cuda(non_blocking=True)

                # Layer-0: same 8th-channel rule as training (--neg_flag)
                _nv = neg_flag_value(name, neg_flag_mode)
                n_noncam_imgs += int(_nv > 0)
                neg_flag = torch.full((1, 1, image.shape[-2], image.shape[-1]),
                                      _nv, device=image.device)
                _, _, _, c4 = model(image, target, aux, neg_flag)
                res = F.interpolate(c4, size=gt_np.shape, mode="bilinear", align_corners=False)
                res = res.sigmoid().data.cpu().numpy().squeeze()
                if use_minmax:      # historical non-standard contrast stretch
                    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                if save_pred_dir:
                    d = os.path.join(save_pred_dir, dname)
                    os.makedirs(d, exist_ok=True)
                    stem = os.path.splitext(os.path.basename(name))[0]
                    save_map(os.path.join(d, stem + ".png"), res)
                mae_sum += np.sum(np.abs(res - gt_np)) * 1.0 / (gt_np.shape[0] * gt_np.shape[1])
            mae = mae_sum / val_loader.size
            per_mae[dname] = mae
            print(f">>> [Val][{dname}] Epoch[{epoch}] MAE={mae:.4f} | "
                  f"prior_mode={prior_mode} split={split} | "
                  f"prior loaded {n_prior_used}/{val_loader.size} "
                  f"(missing {n_prior_missing}, all-zero {n_prior_empty}) | "
                  f"neg_flag=1 on {n_noncam_imgs}/{val_loader.size}")
            logging.info("[Val][{}] Epoch[{}] MAE:{:.4f} prior_mode:{} split:{} "
                         "prior:{}/{} missing:{} allzero:{} noncam:{}/{}"
                         .format(dname, epoch, mae, prior_mode, split, n_prior_used,
                                 val_loader.size, n_prior_missing, n_prior_empty,
                                 n_noncam_imgs, val_loader.size))
            if (prior_mode in ("real", "shuffle") and n_prior_empty
                    and (dname, prior_mode) not in _PRIOR_EMPTY_WARNED):
                _PRIOR_EMPTY_WARNED.add((dname, prior_mode))
                share = 100.0 * n_prior_empty / max(1, n_prior_used)
                print(f">>> [Val][{dname}] WARNING: {n_prior_empty}/{n_prior_used} "
                      f"existing priors are ALL-ZERO ({share:.1f}%): the prior "
                      f"branch is dead for those images (missing-file coverage "
                      f"does not catch this)")
            if strict_prior and n_prior_missing:
                raise SystemExit("ERROR: {} validation image(s) have no real prior "
                                 "under {} (--strict_prior)"
                                 .format(n_prior_missing, prior_root))

    mae = float(np.mean(list(per_mae.values())))
    # A non-finite MAE (diverged epoch: NaN loss -> NaN map) must never win the
    # selection.  The old `epoch == 1` bootstrap clause saved it as "best" on the
    # first epoch (NaN < inf is False, but the clause was unconditional), and the
    # resulting KKK_best_1.pth then made every metric NaN.
    # `best_state["epoch"] == 0` == "nothing selected yet" (a fresh run at --load
    # auto with no best.json), which is the only case that needs the bootstrap.
    if not math.isfinite(mae):
        print(">>> WARNING: epoch {} MAE={} is not finite -> NOT eligible as best"
              .format(epoch, mae))
    elif select_best and (best_state["epoch"] == 0 or mae < best_state["mae"]):
        best_state["mae"] = mae
        best_state["epoch"] = epoch
        path = save_path + "KKK_best_{}.pth".format(epoch)
        # best_epoch == epoch here by construction; record it so a resume from
        # this file does not fall into the old-format warning path
        save_checkpoint(path, model, epoch, best_mae=mae, opt=opt,
                        extra={"best_epoch": int(epoch)})
        # best.json: without it a resumed run starts from mae=inf and overwrites
        # this file on its first epoch, whatever that epoch's MAE is
        write_best(save_path, epoch, mae)
        print(">>> Save best model, epoch {}, MAE={:.4f} -> {}".format(epoch, mae, path))
    elif select_best:
        print(">>> Epoch {} MAE={:.4f}, best={:.4f}@epoch{}".format(
            epoch, mae, best_state["mae"], best_state["epoch"]))
    else:
        print(">>> [final] MAE={:.4f} (best={:.4f}@epoch{})".format(
            mae, best_state["mae"], best_state["epoch"]))
    logging.info("[Val]:Epoch[{}] MAE:{:.4f} Best:{}@epoch{}".format(epoch, mae, best_state["mae"], best_state["epoch"]))
    return per_mae


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train the SAM3 region-prior guided 8-channel network")
    parser.add_argument("--epoch", type=int, default=80)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batchsize", type=int, default=6)
    parser.add_argument("--trainsize", type=int, default=352)
    parser.add_argument("--clip", type=float, default=0.5)
    parser.add_argument("--gpus", type=str, default="0",
                        help="comma-separated GPU IDs, e.g. '0' or '0,1'. More "
                             "than one id only takes effect together with --dp")
    parser.add_argument("--prior_mode", type=str, default="real",
                        choices=["real", "zero", "oracle", "shuffle"],
                        help="ablation: real=SAM3 priors (default), zero=no prior, "
                             "oracle=GT as prior, shuffle=wrong prior")
    parser.add_argument("--img_filter", type=str, default=None,
                        help="keep only training images whose name contains this "
                             "substring (e.g. 'camourflage' for CAMO-only)")
    parser.add_argument("--seed", type=int, default=0, help="global RNG seed")
    parser.add_argument("--minmax", action="store_true",
                        help="apply the historical per-image min-max stretch to the "
                             "validation maps (off by default = standard protocol)")
    parser.add_argument("--strict_prior", action="store_true",
                        help="abort if any validation image has no real prior npz "
                             "(default: report the coverage only)")
    parser.add_argument("--loss_mode", type=str, default="fixed",
                        choices=["fixed", "legacy"],
                        help="'fixed' (default): the Sobel/structure weight really "
                             "scales the BCE term; 'legacy': the pre-2026-10 loss, "
                             "where reduction='mean' collapsed the weight to a scalar "
                             "(keep only to reproduce an old run)")
    parser.add_argument("--neg_flag", type=str, default="label", choices=["label", "zero"],
                        help="8th input channel: 'label' (default) = 1 for COD10K "
                             "'NonCAM' images, i.e. the answer is read from the file "
                             "name; 'zero' = always 0 (use this for every prior/text "
                             "ablation and for any test set that mixes NonCAM images)")
    parser.add_argument("--train_root", type=str, default="./Dataset/TrainValDataset/")
    parser.add_argument("--test_root", type=str, default="./Dataset/TestDataset/")
    parser.add_argument("--test_datasets", type=str, default="CAMO",
                        help="comma-separated test dataset names, e.g. CAMO,COD10K")
    parser.add_argument("--prior_root", type=str, default="./prior_sam3/")
    parser.add_argument("--pretrained_path", type=str,
                        default="./pretrained/pvt_v2_b2.pth",
                        help="ImageNet PVTv2-B2 weights; a missing file aborts "
                             "unless --no_pretrained is given")
    parser.add_argument("--no_pretrained", action="store_true",
                        help="build the backbone from random weights (the log and "
                             "the result would then describe a from-scratch run)")
    parser.add_argument("--save_root", type=str, default="./model_pth/SAM3/")
    parser.add_argument("--load", type=str, default=None,
                        help="resume from checkpoint (e.g. KKK_best_28.pth); "
                             "'auto' = highest epoch under --save_root")
    parser.add_argument("--deterministic", action="store_true",
                        help="cudnn.deterministic=True and cudnn.benchmark=False "
                             "(slower; only kernels with a deterministic "
                             "implementation are bit-exact -- this does not call "
                             "torch.use_deterministic_algorithms)")
    parser.add_argument("--val_split", type=str, default="test",
                        choices=["test", "train"],
                        help="which npz prefix the validation priors come from "
                             "(default: test; train = validate on the *train* npz, "
                             "only meaningful for a train-split val loader)")
    parser.add_argument("--eval_metrics", action="store_true",
                        help="after training, dump predictions of the best epoch and "
                             "run eval_metrics.py (S-measure/E-measure/F-measure/MAE) "
                             "on them, instead of trusting the in-loop MAE alone")
    parser.add_argument("--eval_csv", type=str, default=None,
                        help="CSV appended by --eval_metrics (default: "
                             "<save_root>/eval.csv)")
    parser.add_argument("--workers", type=int, default=8,
                        help="DataLoader worker processes (0 = load in the main "
                             "process, useful to rule out a fork/CUDA deadlock)")
    parser.add_argument("--aug_mode", type=str, default="fast", choices=["fast", "legacy"],
                        help="fast=down-scale to 1.25x trainsize before augmenting "
                             "(default), legacy=augment at native resolution")
    parser.add_argument("--log_interval", type=int, default=10,
                        help="log one [Train] line (with s/step) every N steps")
    parser.add_argument("--mp_context", type=str, default="fork",
                        choices=["fork", "spawn", "forkserver"],
                        help="DataLoader worker start method. fork is the historic "
                             "behaviour; use spawn if a run hangs (CUDA was already "
                             "initialised in the parent before the workers forked)")
    parser.add_argument("--dp", action="store_true",
                        help="enable DataParallel over all --gpus (OFF by "
                             "default: measured 19.06 s/step on 2 GPUs vs "
                             "0.271 s/step on one, ~70x slower)")
    parser.add_argument("--size_rates", type=str, default="0.75,1",
                        help="comma-separated scale factors per step, e.g. "
                             "'0.75,1' (2 forwards/step) or '1' (fastest)")
    parser.add_argument("--amp", action="store_true",
                        help="bf16 autocast for the forward pass (no "
                             "GradScaler needed with bf16); losses stay fp32")
    parser.add_argument("--no_benchmark", action="store_true",
                        help="disable cudnn.benchmark (it re-tunes every new shape, "
                             "which is why the first ~10 steps of a run look 30x slower)")
    return parser


def resolve_gpus(opt):
    """Apply --dp / --gpus and return the effective CUDA_VISIBLE_DEVICES value.

    DataParallel is opt-in: on this box's PCIe-only topology (no NVLink,
    replicate() at ~2.2 GB/s) it is ~70x slower than a single card.
    """
    ids = [g.strip() for g in str(opt.gpus).replace(" ", "").split(",") if g.strip()]
    if not ids:
        ids = ["0"]
    if len(ids) > 1 and not opt.dp:
        print(">>> WARNING: --gpus {} 给了 {} 张卡，但 --dp 未开启 -> 只用 {}"
              .format(opt.gpus, len(ids), ids[0]))
        print(">>>   本机实测：DataParallel 2 卡 19.06 s/step vs 单卡 0.271 s/step "
              "（约 70x 慢）。确需多卡请显式加 --dp。")
        ids = ids[:1]
    return ",".join(ids)


def parse_size_rates(text):
    """'0.75,1' -> [0.75, 1.0]; raises ValueError on anything unusable."""
    rates = []
    for tok in str(text).split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            rate = float(tok)
        except Exception:
            raise ValueError("无法解析 '{}'".format(tok))
        if not (0.2 <= rate <= 4.0):
            raise ValueError("{} 超出 [0.2, 4.0]".format(rate))
        rates.append(rate)
    if not rates:
        raise ValueError("为空")
    return rates


def check_args(opt, n_gpus):
    """Fatal problems caught before any CUDA allocation.  [] == OK."""
    problems = []
    if opt.batchsize < 2:
        problems.append(
            "--batchsize {} 非法：CBAM 的 BatchNorm2d 在 bs=1 时收到 [1,4,1,1]，"
            "会 ValueError，请用 >=2".format(opt.batchsize))
    if n_gpus > 1:
        if opt.batchsize % n_gpus:
            problems.append(
                "--batchsize {} 不能被卡数 {} 整除：torch.chunk 静默丢弃余数，"
                "会有卡空转，请用 {} 的倍数"
                .format(opt.batchsize, n_gpus, n_gpus))
        elif opt.batchsize // n_gpus < 2:
            problems.append(
                "--batchsize {} / {} 卡 = 每卡 {} 张，仍会触发 bs=1 的 BN 崩溃，"
                "至少需要 {}".format(opt.batchsize, n_gpus,
                                     opt.batchsize // n_gpus, 2 * n_gpus))
    try:
        rates = parse_size_rates(opt.size_rates)
        smallest = min(int(round(opt.trainsize * r / 32) * 32) for r in rates)
        if smallest < 64:
            problems.append("--size_rates {} 会得到 {}px 的输入，太小"
                            .format(opt.size_rates, smallest))
    except ValueError as exc:
        problems.append("--size_rates 非法：{}".format(exc))
    return problems


def main(argv=None):
    parser = build_parser()
    opt = parser.parse_args(argv)
    # Trailing separator: roots are used as ``opt.x + "Imgs/"`` / ``+ "KKK_..."``,
    # so ``--save_root model_pth/SAM3`` used to write the files NEXT to the folder
    # (model_pth/SAM3Net_epoch_10.pth) instead of inside it -- silently.
    for _root in ("save_root", "train_root"):
        _v = getattr(opt, _root)
        if _v and not _v.endswith(("/", "\\")):
            setattr(opt, _root, _v + os.sep)

    opt.gpus = resolve_gpus(opt)
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    cudnn.benchmark = not opt.no_benchmark

    problems = check_args(opt, len([g for g in opt.gpus.split(",") if g]))
    if problems:
        for msg in problems:
            print(">>> [FATAL] " + msg)
        return 2
    size_rates = parse_size_rates(opt.size_rates)
    print(f">>> size_rates={size_rates}  amp={opt.amp}  dp={opt.dp}")

    # ---- reproducibility ----
    # set_seed() also enables cudnn.deterministic when --deterministic is given;
    # formerly only the four RNGs were seeded and the workers inherited nothing.
    cudnn_state = set_seed(opt.seed, deterministic=opt.deterministic,
                           benchmark=(not opt.no_benchmark))
    print(f">>> seed={opt.seed}  prior_mode={opt.prior_mode}  minmax={opt.minmax}  "
          f"cudnn.benchmark={cudnn_state['cudnn.benchmark']}  "
          f"deterministic={cudnn_state['cudnn.deterministic']}")

    model = Network(channel=64, imagenet_pretrained=not opt.no_pretrained,
                    pretrained_path=opt.pretrained_path).cuda()
    # ---- amp sanity: bf16 needs Ampere or newer ----
    if opt.amp and not torch.cuda.is_bf16_supported():
        print(">>> WARNING: 当前 GPU 不支持 bf16，--amp 已自动关闭")
        opt.amp = False
    print(f">>> amp: {'bf16 autocast' if opt.amp else 'off'}")
    # ---- multi-GPU DataParallel (opt-in via --dp) ----
    # resolve_gpus() already narrowed CUDA_VISIBLE_DEVICES to a single card
    # when --dp is off, so n_gpus > 1 only happens on an explicit opt-in.
    n_gpus = torch.cuda.device_count()
    if n_gpus > 1 and opt.dp:
        model = torch.nn.DataParallel(model)
        print(f">>> DataParallel: using {n_gpus} GPUs ({opt.gpus}) "
              f"[--dp given: expect ~70x slowdown on PCIe-only boxes]")
    else:
        print(f">>> Single GPU ({opt.gpus})")
    optimizer = optim.AdamW(model.parameters(), opt.lr)
    save_path = opt.save_root
    os.makedirs(save_path, exist_ok=True)

    # ---- resume from checkpoint ----
    import re
    start_epoch = 1
    best_state = {"mae": float("inf"), "epoch": 0}
    meta = {}          # checkpoint metadata; stays empty when not resuming
    if opt.load:
        path = opt.load
        if path.lower() == "auto":
            path = find_latest_checkpoint(save_path)
            if not path:
                print(">>> [FATAL] --load auto: no *.pth under " + save_path)
                return 2
            print(f">>> --load auto -> {path}")
        print(f">>> Loading checkpoint: {path}")
        # Layer-0 guard: a 7ch / fusion-less file must not resume silently.
        # load_checkpoint only reads the file once (the check inspects the same dict).
        ckpt, meta = load_checkpoint(path)
        assert_state_dict_channels(ckpt, expected=8, source=path)
        ckpt = {(k[len("module."):] if k.startswith("module.") else k): v
                for k, v in ckpt.items()}
        save_model = model.module if hasattr(model, 'module') else model
        missing, unexpected = save_model.load_state_dict(ckpt, strict=False)
        if missing or unexpected:
            raise ValueError("{}: {} missing / {} unexpected key(s) -> the "
                             "checkpoint is not this architecture (e.g. missing {}; "
                             "unexpected {})".format(path, len(missing), len(unexpected),
                                                     missing[:3], unexpected[:3]))
        # epoch from the checkpoint dict, not from the file name: a copied or
        # renamed checkpoint keeps its epoch instead of restarting at 1
        if meta.get("epoch") is not None:
            start_epoch = int(meta["epoch"]) + 1
        else:
            m = re.search(r"(?:epoch_|KKK_best_)(\d+)", os.path.basename(path))
            start_epoch = (int(m.group(1)) + 1) if m else 1
            print(">>> WARNING: {} has no 'epoch' -> start_epoch={} was guessed from "
                  "the file name (rename it and resume restarts at 1)"
                  .format(path, start_epoch))
        # best MAE must survive the restart: with inf here, the first resumed epoch
        # always overwrote KKK_best_*.pth even when it was worse
        best = read_best(save_path)
        if best is None and meta.get("best_mae") is not None:
            # best_epoch is the epoch whose weights are in KKK_best_<epoch>.pth; it
            # is NOT necessarily the epoch this snapshot was taken at.  Snapshots
            # written by this version carry it explicitly; an older file does not.
            best_epoch = meta.get("best_epoch")
            if best_epoch is None:
                best_epoch = meta.get("epoch") or 0
                print(">>> WARNING: {} has best_mae but no best_epoch (old format) "
                      "-> assuming epoch {} is the best one; --eval_metrics will "
                      "refuse if KKK_best_{}.pth does not exist"
                      .format(path, best_epoch, best_epoch))
            best = {"mae": float(meta["best_mae"]), "epoch": int(best_epoch)}
            write_best(save_path, best["epoch"], best["mae"])
            print(">>> best.json was missing; restored from the checkpoint dict")
        if best is None:
            print(">>> WARNING: no best.json and no best_mae in {} -> best_state=inf, "
                  "the first validation of this run will set a new best".format(path))
        else:
            best_state = best
            best_file = save_path + "KKK_best_{}.pth".format(best_state["epoch"])
            if not os.path.isfile(best_file):
                print(">>> WARNING: best.json points at epoch {} but {} is missing; "
                      "selection continues from MAE={:.4f} and the file will only "
                      "be re-created if a later epoch is better"
                      .format(best_state["epoch"], best_file, best_state["mae"]))
        print(f">>> Resumed from {path}, continue at epoch {start_epoch}, "
              f"best MAE={best_state['mae']:.4f}@epoch{best_state['epoch']}")

    train_loader = get_sam3_loader(
        image_root=opt.train_root + "Imgs/", gt_root=opt.train_root + "GT/",
        prior_root=opt.prior_root, batchsize=opt.batchsize, trainsize=opt.trainsize,
        split="train", num_workers=opt.workers, prior_mode=opt.prior_mode,
        img_filter=opt.img_filter, seed=opt.seed, aug_mode=opt.aug_mode,
        neg_flag_mode=opt.neg_flag,
        mp_context=(opt.mp_context if opt.workers > 0 else None))
    print(f">>> loader: workers={opt.workers} aug_mode={opt.aug_mode} "
          f"aug_size={train_loader.dataset.aug_size} "
          f"batchsize={opt.batchsize} steps/epoch={len(train_loader)}")

    # ---- Layer-0 guard: a dead prior branch must be impossible to miss ----
    # (the historic SAM3 run used a prior_root that did not exist and trained on
    #  all-zero priors without a single warning)
    try:
        check_prior_coverage(opt.prior_root, opt.train_root + "Imgs/", split="train",
                             img_filter=opt.img_filter,
                             strict=(opt.prior_mode == "real"))
    except FileNotFoundError as exc:
        # strict=True raises by design; report it as a normal non-zero exit
        # instead of a traceback (the script's other fatal paths return 2)
        print(">>> [FATAL] " + str(exc))
        return 2

    # build validation loaders from one or more test datasets
    val_loaders = []
    for _dname in [s.strip() for s in opt.test_datasets.split(",") if s.strip()]:
        _dp = os.path.join(opt.test_root, _dname)
        if not (os.path.isdir(os.path.join(_dp, "Imgs"))
                and os.path.isdir(os.path.join(_dp, "GT"))):
            print(f">>> WARNING: test dataset not found, skip: {_dp}")
            continue
        val_loaders.append((
            _dname,
            # trailing '/' (not os.sep): dataloader.test_dataset derives the file
            # name with path.split('/')[-1], so an os.sep root would leak the full
            # path into `name` and break every prior lookup
            test_dataset(image_root=os.path.join(_dp, "Imgs").replace(os.sep, "/") + "/",
                         gt_root=os.path.join(_dp, "GT").replace(os.sep, "/") + "/",
                         testsize=opt.trainsize)))
    assert val_loaders, f"no valid test dataset under {opt.test_root}"
    total_step = len(train_loader)

    logging.basicConfig(filename=save_path + "log.log", format="[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]",
                        level=logging.INFO, filemode="a", datefmt="%Y-%m-%d %I:%M:%S %p")
    logging.info(">>> config: {}".format(opt))
    print(">>> config:", opt)
    # provenance next to the weights: log.log is opened in append mode, so two
    # runs in the same --save_root were indistinguishable afterwards
    write_run_manifest(os.path.join(save_path, "run_manifest.json"), opt,
                       extra={"cudnn": cudnn_state,
                              "train_images": len(train_loader.dataset),
                              "val_datasets": [d for d, _ in val_loaders],
                              # what a resumed run inherited (best.json at launch)
                              "start_epoch": start_epoch,
                              "best_at_start": dict(best_state)})

    # CosineAnnealingLR with last_epoch >= 0 (i.e. every resume) requires
    # 'initial_lr' in each param group, otherwise torch raises
    # "param 'initial_lr' is not specified in param_groups[0] when resuming an
    # optimizer" -- which aborted every --load run (Layer-0, 2026-10).  The
    # checkpoint carries weights only, so the optimizer genuinely starts fresh
    # and --lr is the correct base.
    for group in optimizer.param_groups:
        group.setdefault("initial_lr", group["lr"])
    if meta.get("args") and int(meta["args"].get("epoch", opt.epoch)) != opt.epoch:
        print(">>> WARNING: checkpoint was written for --epoch {} but this run "
              "uses --epoch {} -> T_max changes, so the cosine schedule is NOT "
              "the one the run started with; pass the original --epoch to "
              "continue it".format(meta["args"]["epoch"], opt.epoch))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, opt.epoch), eta_min=1e-6,
        last_epoch=start_epoch - 2)   # resume: continue the same schedule

    for epoch in range(start_epoch, opt.epoch + 1):  # 闭区间：--epoch N 训练到第 N 轮
        logging.info(">>> current lr: {}".format(scheduler.get_last_lr()[0]))
        train(train_loader, model, optimizer, epoch, size_rates, total_step, opt)
        test(val_loaders, model, epoch, save_path, best_state, opt.prior_root,
             use_minmax=opt.minmax, strict_prior=opt.strict_prior,
             prior_mode=opt.prior_mode, neg_flag_mode=opt.neg_flag,
             split=opt.val_split, opt=opt)
        scheduler.step()   # PyTorch 1.1+: step AFTER optimizer, not before
        if epoch % 10 == 0:
            # Snapshot AFTER validation so it records the run's real best epoch
            # (best_epoch) -- a snapshot taken inside train() could only write
            # best_mae=None, which made resume guesswork.
            path = save_path + "Net_epoch_{}.pth".format(epoch)
            save_checkpoint(path, model, epoch, best_mae=best_state["mae"], opt=opt,
                            extra={"best_epoch": int(best_state["epoch"])})
            print(">>> Saved", path)

    if opt.eval_metrics:
        return final_eval(val_loaders, model, save_path, best_state, opt)
    return 0


def final_eval(val_loaders, model, save_path, best_state, opt):
    """Score the *best* checkpoint with eval_metrics.py on the same loaders.

    The in-loop MAE is measured at ``--trainsize`` and (historically) after an
    optional min-max stretch, so it cannot be compared with a number produced by
    ``MyTest.py`` + ``eval_metrics.py``.  This pass loads KKK_best_<epoch>.pth,
    writes <save_root>/pred_metrics/<dataset>/<stem>.png and appends the
    4-metric row to <save_root>/eval.csv, so the reported number comes from the
    exact weights best.json points at.
    """
    epoch = int(best_state["epoch"])
    path = save_path + "KKK_best_{}.pth".format(epoch)
    if not os.path.isfile(path):
        print(">>> [FATAL] --eval_metrics: {} not found (no best epoch was saved)"
              .format(path))
        return 2
    state, _ = load_checkpoint(path)
    assert_state_dict_channels(state, expected=8, source=path)
    net = model.module if hasattr(model, 'module') else model
    net.load_state_dict(state)
    pred_root = os.path.join(save_path, "pred_metrics")
    # Regenerate predictions from scratch: a leftover PNG from an earlier best
    # epoch (or from an earlier --test_datasets list) has no matching GT, and
    # eval_metrics.py aborts on n_pred > n_gt -- or, if the stem does exist,
    # scores a map this forward pass never produced.
    if os.path.isdir(pred_root):
        shutil.rmtree(pred_root)
        print(">>> [eval] cleared stale predictions: {}".format(pred_root))
    test(val_loaders, model, epoch, save_path, best_state, opt.prior_root,
         use_minmax=False, strict_prior=opt.strict_prior,
         prior_mode=opt.prior_mode, neg_flag_mode=opt.neg_flag,
         split=opt.val_split, save_pred_dir=pred_root, select_best=False, opt=opt)
    entries = []
    gt_exts = set()
    for dname, loader in val_loaders:
        ds = loader.dataset if hasattr(loader, "dataset") else loader
        gt_root = getattr(ds, "gt_root", None)
        if gt_root is None:
            print(">>> [FATAL] cannot find gt_root on the {} loader".format(dname))
            return 2
        # GT is not always .png (test_dataset accepts .tif/.tiff/.jpg); the stem
        # of the prediction still matches, only --gt_ext has to be right.
        gt_exts |= {os.path.splitext(g)[1].lower() for g in getattr(ds, "gts", [])}
        entries.append({"name": "{}_{}".format(dname, epoch),
                        "pred_dir": os.path.join(pred_root, dname),
                        "gt_dir": gt_root.rstrip("/\\"),
                        "dataset": dname,
                        "split": opt.val_split,
                        "pred_protocol": "raw"})
    manifest = write_eval_manifest(os.path.join(save_path, "eval_manifest.json"),
                                   entries)
    extra = []
    if len(gt_exts) == 1:
        extra = ["--gt_ext", gt_exts.pop()]
    elif gt_exts:
        print(">>> [eval] WARNING: GT extensions differ across datasets ({}); "
              "eval_metrics.py only accepts one --gt_ext, falling back to .png"
              .format(", ".join(sorted(gt_exts))))
    csv = opt.eval_csv or os.path.join(save_path, "eval.csv")
    return run_eval_metrics(manifest, csv=csv, group_by="dataset", extra_args=extra)


if __name__ == "__main__":
    sys.exit(main())
