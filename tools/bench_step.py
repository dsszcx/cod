#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/bench_step.py - 训练速度分段压测（先定位"慢"在哪一段，再谈优化）

四段，口径与 MyTrain_SAM3 的训练循环一致：
  data : DataLoader 出一个 batch（含增强 + 先验 npz 读取 + 排列）
  rates: H2D + 各 size_rate 的 interpolate（搬运/缩放开销）
  step : H2D + interpolate + 前向 + 4 路深监督损失 + backward + step
  full : data + step（= 日志里 s/step 的口径）

典型用法（服务器，cd 到仓库根目录）：
  python tools/bench_step.py --prior_root ./prior_sam3_c448 --batchsize 6 --workers 8
  python tools/bench_step.py --mode step --synthetic --batchsize 6
  python tools/bench_step.py --gpus 0,1 --dp --synthetic      # 复现 DP 的慢

判读：
  * data 占大头  -> 增强 / 先验 IO 慢：用 --aug_mode fast、压实先验、加 --workers
  * step 占大头  -> 算力瓶颈：--amp、加 --batchsize、--size_rates 1
  * 单卡 step 只有 0.1 s 量级、加上 --dp 立刻变 ~9 s -> PCIe-only 机器的
    DataParallel 开销（实测 70x），别开 --dp
"""
import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from MyTrain_SAM3 import (hybrid_e_loss, parse_size_rates,  # noqa: E402
                          resolve_gpus)
from Network_SAM3 import Network  # noqa: E402
from utils import clip_gradient  # noqa: E402


def build_parser():
    p = argparse.ArgumentParser(description="训练速度分段压测")
    p.add_argument("--mode", default="full",
                   choices=["data", "rates", "step", "full"])
    p.add_argument("--iters", type=int, default=20, help="计时迭代次数")
    p.add_argument("--warmup", type=int, default=3, help="预热次数（cudnn autotune）")
    p.add_argument("--synthetic", action="store_true",
                   help="用随机张量代替数据集（不读盘，只测 GPU 侧）")
    # 与 MyTrain_SAM3 同名同义
    p.add_argument("--gpus", default="0")
    p.add_argument("--dp", action="store_true")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--batchsize", type=int, default=6)
    p.add_argument("--trainsize", type=int, default=352)
    p.add_argument("--size_rates", default="0.75,1")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--aug_mode", default="fast", choices=["fast", "legacy"])
    p.add_argument("--mp_context", default="fork",
                   choices=["fork", "spawn", "forkserver"])
    p.add_argument("--prior_root", default="./prior_sam3/")
    p.add_argument("--train_root", default="./Dataset/TrainValDataset/")
    p.add_argument("--img_filter", default=None)
    return p


def make_loader(opt):
    from dataloader_sam3 import get_sam3_loader
    return get_sam3_loader(
        image_root=opt.train_root + "Imgs/", gt_root=opt.train_root + "GT/",
        prior_root=opt.prior_root, batchsize=opt.batchsize,
        trainsize=opt.trainsize, split="train", num_workers=opt.workers,
        prior_mode="real", img_filter=opt.img_filter, seed=0,
        aug_mode=opt.aug_mode,
        mp_context=(opt.mp_context if opt.workers > 0 else None))


def synthetic_batch(opt):
    b, s = opt.batchsize, opt.trainsize
    return (torch.rand(b, 3, s, s),
            torch.randint(0, 2, (b, 1, s, s)).float(),
            torch.randint(0, 2, (b, 4, s, s)).float(),
            torch.randint(0, 2, (b, 4, s, s)).float(),
            torch.zeros(b, 1, s, s))


def main(argv=None):
    parser = build_parser()
    opt = parser.parse_args(argv)
    opt.gpus = resolve_gpus(opt)
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    torch.backends.cudnn.benchmark = True

    if opt.mode == "data" and opt.synthetic:
        print(">>> --mode data 需要真实数据集，不能和 --synthetic 同用")
        return 2
    if opt.batchsize < 2:
        print(">>> --batchsize 必须 >= 2（CBAM 的 BN 在 bs=1 会崩）")
        return 2

    size_rates = parse_size_rates(opt.size_rates)
    iters, warmup = max(1, opt.iters), max(0, opt.warmup)

    def timed(fn, label):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        dt = (time.time() - t0) / iters
        print(">>> %-6s %8.4f s/iter  %7.2f img/s"
              % (label, dt, opt.batchsize / dt if dt > 0 else 0.0))
        return dt

    # ---- 取数 ----
    loader, loader_it = None, None
    if not opt.synthetic:
        loader = make_loader(opt)
        print(">>> loader: workers={} steps={}".format(opt.workers, len(loader)))
        loader_it = iter(loader)

    def next_batch():
        nonlocal loader_it
        if opt.synthetic:
            return synthetic_batch(opt)
        try:
            return next(loader_it)
        except StopIteration:
            loader_it = iter(loader)
            return next(loader_it)

    print(">>> gpus={} dp={} amp={} bs={} ts={} size_rates={}"
          .format(opt.gpus, opt.dp, opt.amp, opt.batchsize, opt.trainsize,
                  size_rates))

    if opt.mode == "data":
        timed(next_batch, "data")
        return 0

    model = Network(channel=64).cuda()
    if opt.dp and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(">>> DataParallel over {} GPUs".format(torch.cuda.device_count()))
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), 2e-4)
    cpu_batch = next_batch()

    def move(cpu):
        return [t.cuda(non_blocking=True) for t in cpu]

    def sized(images, gts, targets, auxs, neg_flags, rate):
        s = max(32, int(round(opt.trainsize * rate / 32) * 32))
        return (F.interpolate(images, size=(s, s), mode="bicubic", align_corners=True),
                F.interpolate(gts, size=(s, s), mode="bicubic", align_corners=True),
                F.interpolate(targets, size=(s, s), mode="bilinear", align_corners=False),
                F.interpolate(auxs, size=(s, s), mode="bilinear", align_corners=False),
                F.interpolate(neg_flags, size=(s, s), mode="nearest"))

    def h2d_only():
        images, gts, targets, auxs, neg_flags = move(cpu_batch)
        for rate in size_rates:
            sized(images, gts, targets, auxs, neg_flags, rate)

    def one_step(cpu):
        images, gts, targets, auxs, neg_flags = move(cpu)
        for rate in size_rates:
            im, gt, tg, ax, nf = sized(images, gts, targets, auxs, neg_flags, rate)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=opt.amp):
                edge, c2, c3, c4 = model(im, tg, ax, nf)
            loss = (0.4 * hybrid_e_loss(edge.float(), gt)
                    + 0.25 * hybrid_e_loss(c2.float(), gt)
                    + 0.2 * hybrid_e_loss(c3.float(), gt)
                    + 0.15 * hybrid_e_loss(c4.float(), gt))
            loss.backward()
            clip_gradient(optimizer, 0.5)
            optimizer.step()

    step_dt = None
    if opt.mode == "rates":
        timed(h2d_only, "rates")
    elif opt.mode in ("step", "full"):
        step_dt = timed(lambda: one_step(cpu_batch), "step")

    if opt.mode == "full":
        def full():
            nonlocal cpu_batch
            cpu_batch = next_batch()
            one_step(cpu_batch)
        full_dt = timed(full, "full")
        if step_dt:
            data_dt = max(full_dt - step_dt, 0.0)
            print(">>> 拆解: data~%.4f s + step~%.4f s = %.4f s/iter"
                  % (data_dt, step_dt, full_dt))
            if step_dt > 1.0 and step_dt > 10 * max(data_dt, 1e-6):
                print(">>> 提示: step 占绝对多数。若单卡 step 只有 0.1 s 量级，"
                      "说明踩到了 DataParallel（去掉 --dp）。")

    print(">>> 健康值: 单卡 2 个 size_rate 约 0.2~0.35 s/step；"
          "DP 2 卡实测 19 s/step（约 70x）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
