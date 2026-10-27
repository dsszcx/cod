"""verify_pipeline.py - Smoke-test the CAMO+COD10K training pipeline.

Checks (no GPU needed for parts 1-2):
  1. merged train dataset (CAMO + COD10K) loads with correct sample count
  2. a few batches produce correctly shaped (image, gt, target, aux)
  3. CAMO test / COD10K test loaders build correctly
  4. if CUDA is available: one forward+backward step of the 8-channel network

Usage:
  python verify_pipeline.py
"""
import argparse

import torch

from dataloader import test_dataset
from dataloader_sam3 import get_sam3_loader


def check_train(opt):
    print("=" * 60)
    print("[1/4] merged train loader (CAMO + COD10K)")
    loader = get_sam3_loader(
        image_root=opt.train_root + "Imgs/",
        gt_root=opt.train_root + "GT/",
        prior_root=opt.prior_root,
        batchsize=opt.batchsize, trainsize=opt.trainsize,
        split="train", num_workers=0)
    print(f"    dataset size: {loader.dataset.size}")
    assert loader.dataset.size == opt.expected_train, \
        f"expected {opt.expected_train} samples, got {loader.dataset.size}"

    for i, (images, gts, targets, auxs, neg_flags) in enumerate(loader):
        print(f"    batch {i}: image {tuple(images.shape)} gt {tuple(gts.shape)} "
              f"target {tuple(targets.shape)} aux {tuple(auxs.shape)}")
        assert images.shape[1:] == (3, opt.trainsize, opt.trainsize)
        assert gts.shape[1:] == (1, opt.trainsize, opt.trainsize)
        assert targets.shape[1:] == (1, opt.trainsize, opt.trainsize)
        assert auxs.shape[1:] == (3, opt.trainsize, opt.trainsize)
        assert targets.min() >= 0 and targets.max() <= 1
        assert auxs.min() >= 0 and auxs.max() <= 1
        if i == 2:
            break
    print("    OK")


def check_test(opt):
    print("=" * 60)
    for dname, expected in (("CAMO", opt.expected_camo), ("COD10K", opt.expected_cod10k)):
        dp = opt.test_root + dname
        loader = test_dataset(dp + "/Imgs/", dp + "/GT/", opt.trainsize)
        print(f"[2/4] test loader {dname}: size={loader.size}")
        assert loader.size == expected, f"{dname}: expected {expected}, got {loader.size}"
    print("    OK")


def check_network(opt):
    print("=" * 60)
    print("[3/4] 8-channel network forward + backward")
    if not torch.cuda.is_available():
        print("    CUDA not available, skipping network step")
        return
    from Network_SAM3 import Network
    model = Network(channel=64, imagenet_pretrained=True).cuda().train()
    opt_ = torch.optim.AdamW(model.parameters(), lr=1e-5)
    for _ in range(2):
        images = torch.randn(2, 3, 352, 352).cuda()
        gts = (torch.rand(2, 1, 352, 352) > 0.5).float().cuda()
        targets = (torch.rand(2, 1, 352, 352) > 0.5).float().cuda()
        auxs = (torch.rand(2, 3, 352, 352) > 0.5).float().cuda()
        opt_.zero_grad()
        neg_flags = torch.zeros(2, 1, 352, 352).cuda()
        outs = model(images, targets, auxs, neg_flags)
        loss = sum(o.mean() for o in outs)
        loss.backward()
        opt_.step()
        print(f"    step: outputs={[tuple(o.shape) for o in outs]}, loss={loss.item():.4f}")
    print("    OK")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_root", type=str, default="./Dataset/TrainValDataset/")
    parser.add_argument("--test_root", type=str, default="./Dataset/TestDataset/")
    parser.add_argument("--prior_root", type=str, default="./prior_sam3/")
    parser.add_argument("--trainsize", type=int, default=352)
    parser.add_argument("--batchsize", type=int, default=4)
    parser.add_argument("--expected_train", type=int, default=7000,
                        help="expected merged train size (CAMO 1000 + COD10K all 6000)")
    parser.add_argument("--expected_camo", type=int, default=250)
    parser.add_argument("--expected_cod10k", type=int, default=4000,
                        help="expected COD10K test size (CAM 2026 + NONCAM 1974)")
    opt = parser.parse_args()

    check_train(opt)
    check_test(opt)
    check_network(opt)
    print("=" * 60)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
