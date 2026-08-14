"""MyTrain_Prior.py - Train the LLM-prior-guided network (Network_Prior.py).

Usage:
  python MyTrain_Prior.py --epoch 50 --batchsize 8 --vocab_size 256 --prior_root ./prior/
"""
import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch import optim
from tqdm import tqdm

import logging
from dataloader_prior import get_prior_loader
from dataloader import test_dataset
from utils import clip_gradient, AvgMeter
from Network_Prior import Network


# ---------- loss (same as original) ----------
def hybrid_e_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction="mean")
    wbce = ((weit * wbce).sum(dim=(2, 3)) + 1e-8) / (weit.sum(dim=(2, 3)) + 1e-8)
    pred = torch.sigmoid(pred)
    mpred = pred.mean(dim=(2, 3)).view(pred.shape[0], pred.shape[1], 1, 1).repeat(1, 1, pred.shape[2], pred.shape[3])
    phiFM = pred - mpred
    mmask = mask.mean(dim=(2, 3)).view(mask.shape[0], mask.shape[1], 1, 1).repeat(1, 1, mask.shape[2], mask.shape[3])
    phiGT = mask - mmask
    EFM = (2.0 * phiFM * phiGT + 1e-8) / (phiFM * phiFM + phiGT * phiGT + 1e-8)
    QFM = (1 + EFM) * (1 + EFM) / 4.0
    eloss = 1.0 - QFM.mean(dim=(2, 3))
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1.0 - (inter + 1 + 1e-8) / (union - inter + 1 + 1e-8)
    return (wbce + eloss + wiou).mean()


# ---------- training ----------
def train(train_loader, model, optimizer, epoch, size_rates, total_step, opt):
    model.train()
    loss_all = 0
    epoch_step = 0
    loss_g_record, loss_c_record = AvgMeter(), AvgMeter()
    pbar = tqdm(enumerate(train_loader, start=1), total=len(train_loader),
                desc=f"Epoch {epoch:03d}/{opt.epoch:03d}", ncols=100)
    for i, (images, gts, pos_maps, tag_bows) in pbar:
        for rate in size_rates:
            optimizer.zero_grad()
            images = images.cuda()
            gts = gts.cuda()
            pos_maps = pos_maps.cuda()                     # already (B,1,H,W)
            tag_bows = tag_bows.cuda()                # (B,V)

            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.interpolate(images, size=(trainsize, trainsize), mode="bicubic", align_corners=True)
                gts = F.interpolate(gts, size=(trainsize, trainsize), mode="bicubic", align_corners=True)
                pos_maps = F.interpolate(pos_maps, size=(trainsize, trainsize), mode="bilinear", align_corners=True)

            edge, c2, c3, c4 = model(images, pos_maps, tag_bows)
            loss_g = hybrid_e_loss(edge, gts)
            loss_c = hybrid_e_loss(c2, gts) + hybrid_e_loss(c3, gts) + hybrid_e_loss(c4, gts)
            loss = loss_g + loss_c

            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            epoch_step += 1
            loss_all += loss.data
            if rate == 1:
                loss_g_record.update(loss_g.data, opt.batchsize)
                loss_c_record.update(loss_c.data, opt.batchsize)

        pbar.set_postfix({"loss_g": f"{loss_g_record.show():.4f}", "loss_c": f"{loss_c_record.show():.4f}"})
        if i % 10 == 0 or i == total_step:
            logging.info("[Train]:Epoch[{:03d}/{:03d}] Step[{:04d}/{:04d}] loss_g:{:.4f} loss_c:{:.4f}".format(
                epoch, opt.epoch, i, total_step, loss_g_record.show(), loss_c_record.show()))

    loss_all /= epoch_step
    logging.info("[Train]:Epoch[{:03d}/{:03d}] Loss_AVG:{:.4f}".format(epoch, opt.epoch, loss_all))
    if epoch % 10 == 0:
        torch.save(model.state_dict(), opt.save_root + "Net_epoch_{}.pth".format(epoch))
        print(">>> Saved", opt.save_root + "Net_epoch_{}.pth".format(epoch))


# ---------- validation on test set with priors ----------
def test(val_loader, model, epoch, save_path, best_state, prior_root, vocab_size):
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        for i in range(val_loader.size):
            _, image, gt, name, _ = val_loader.load_data()
            img_name = name.replace(".png", ".jpg")
            npz_name = f"test_{img_name.replace('.', '_')}.npz"
            npz_path = os.path.join(prior_root, npz_name)
            if os.path.exists(npz_path):
                d = np.load(npz_path)
                pos_map = torch.from_numpy(d["pos_map"]).float()
                tag_bow = torch.from_numpy(d["tag_bow"]).float()
            else:
                pos_map = torch.zeros(1, 352, 352)
                tag_bow = torch.zeros(vocab_size)
            if pos_map.shape[-1] != image.shape[-1]:
                pos_map = F.interpolate(pos_map.unsqueeze(0), size=image.shape[-2:],
                                        mode="bilinear", align_corners=False).squeeze(0)
            gt_np = np.asarray(gt, np.float32) / 255.0
            image = image.cuda()
            pos_map = pos_map.unsqueeze(0).cuda()
            tag_bow = tag_bow.unsqueeze(0).cuda()

            _, _, _, c4 = model(image, pos_map, tag_bow)
            res = F.interpolate(c4, size=gt_np.shape, mode="bilinear", align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res - gt_np)) * 1.0 / (gt_np.shape[0] * gt_np.shape[1])
        mae = mae_sum / val_loader.size

    if epoch == 1 or mae < best_state["mae"]:
        best_state["mae"] = mae
        best_state["epoch"] = epoch
        torch.save(model.state_dict(), save_path + "KKK_best_{}.pth".format(epoch))
        print(">>> Save best model, epoch {}, MAE={:.4f}".format(epoch, mae))
    else:
        print(">>> Epoch {} MAE={:.4f}, best={:.4f}@epoch{}".format(
            epoch, mae, best_state["mae"], best_state["epoch"]))
    logging.info("[Val]:Epoch[{}] MAE:{:.4f} Best:{}@epoch{}".format(epoch, mae, best_state["mae"], best_state["epoch"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batchsize", type=int, default=8)
    parser.add_argument("--trainsize", type=int, default=352)
    parser.add_argument("--clip", type=float, default=0.5)
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--train_root", type=str, default="./Dataset/TrainValDataset/")
    parser.add_argument("--test_root", type=str, default="./Dataset/TestDataset/CAMO/")
    parser.add_argument("--prior_root", type=str, default="./prior/")
    parser.add_argument("--vocab_size", type=int, default=256)
    parser.add_argument("--save_root", type=str, default="./model_pth/PRIOR/")
    opt = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
    cudnn.benchmark = True

    model = Network(channel=64, vocab_size=opt.vocab_size).cuda()
    optimizer = optim.AdamW(model.parameters(), opt.lr)
    save_path = opt.save_root
    os.makedirs(save_path, exist_ok=True)

    train_loader = get_prior_loader(
        image_root=opt.train_root + "Imgs/", gt_root=opt.train_root + "GT/",
        prior_root=opt.prior_root, batchsize=opt.batchsize, trainsize=opt.trainsize,
        vocab_size=opt.vocab_size, num_workers=4)
    val_loader = test_dataset(image_root=opt.test_root + "Imgs/", gt_root=opt.test_root + "GT/",
                              testsize=opt.trainsize)
    total_step = len(train_loader)

    logging.basicConfig(filename=save_path + "log.log", format="[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]",
                        level=logging.INFO, filemode="a", datefmt="%Y-%m-%d %I:%M:%S %p")
    logging.info(">>> config: {}".format(opt))
    print(">>> config:", opt)

    best_state = {"mae": 1, "epoch": 0}
    size_rates = [0.75, 1]  # fit 8GB VRAM
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-5)

    for epoch in range(1, opt.epoch):
        scheduler.step()
        logging.info(">>> current lr: {}".format(scheduler.get_last_lr()[0]))
        train(train_loader, model, optimizer, epoch, size_rates, total_step, opt)
        test(val_loader, model, epoch, save_path, best_state, opt.prior_root, opt.vocab_size)