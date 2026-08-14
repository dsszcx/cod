"""MyTest_SAM3.py - Inference with SAM3 region-prior network.

Usage:
  python MyTest_SAM3.py --pth_path ./model_pth/SAM3/KKK_best_2.pth --prior_root ./prior_sam3/
"""
import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import cv2
from dataloader import test_dataset
from Network_SAM3 import Network


def load_test_prior(name, prior_root, hw):
    img_name = name.replace(".png", ".jpg")
    npz_name = f"test_{img_name.replace('.', '_')}.npz"
    npz_path = os.path.join(prior_root, npz_name)
    if os.path.exists(npz_path):
        d = np.load(npz_path)
        target = torch.from_numpy(d["target_mask"]).float()
        aux = torch.from_numpy(d["aux_masks"]).float()
    else:
        target = torch.zeros(1, 352, 352)
        aux = torch.zeros(3, 352, 352)
    if target.shape[-1] != hw[-1]:
        target = F.interpolate(target.unsqueeze(0), size=hw, mode="bilinear", align_corners=False).squeeze(0)
        aux = F.interpolate(aux.unsqueeze(0), size=hw, mode="bilinear", align_corners=False).squeeze(0)
    return target, aux


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--testsize", type=int, default=352)
    parser.add_argument("--pth_path", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="./Dataset/TestDataset/")
    parser.add_argument("--save_root", type=str, default="./results_sam3/")
    parser.add_argument("--prior_root", type=str, default="./prior_sam3/")
    parser.add_argument("--gpu_id", type=str, default="0")
    opt = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
    cudnn.benchmark = True

    model = Network(channel=64).cuda()
    model.load_state_dict(torch.load(opt.pth_path), strict=False)
    model.eval()

    txt_save_path = os.path.join(opt.save_root, os.path.basename(os.path.dirname(opt.pth_path)))
    os.makedirs(txt_save_path, exist_ok=True)
    print(">>> config:", opt)

    for _data_name in ["CAMO"]:
        map_save_path = txt_save_path + "/{}/".format(_data_name)
        os.makedirs(map_save_path, exist_ok=True)
        data_path = os.path.join(opt.data_root, _data_name)
        test_loader = test_dataset("{}/Imgs/".format(data_path), "{}/GT/".format(data_path), opt.testsize)

        with torch.no_grad():
            for i in range(test_loader.size):
                _, image, gt, name, _ = test_loader.load_data()
                gt = np.asarray(gt, np.float32)
                image = image.cuda()
                target, aux = load_test_prior(name, opt.prior_root, image.shape[-2:])
                target = target.unsqueeze(0).cuda()
                aux = aux.unsqueeze(0).cuda()

                _, _, _, c4 = model(image, target, aux)
                res = F.interpolate(c4, size=gt.shape, mode="bilinear", align_corners=False)
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                cv2.imwrite(map_save_path + name, res * 255)
                print(">>> saved:", map_save_path + name)