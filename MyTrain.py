import argparse
import numpy as np
from datetime import datetime
from torch import optim
import torch
from dataloader import get_loader, test_dataset
from utils import clip_gradient, AvgMeter
import torch.nn.functional as F
import os

import logging
import torch.backends.cudnn as cudnn
from Network_PVTv2 import Network
from tqdm import tqdm




def hybrid_e_loss(pred, mask):
    """ Hybrid Eloss """
    # adaptive weighting masks
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)

    # weighted binary cross entropy loss function
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')
    wbce = ((weit * wbce).sum(dim=(2, 3)) + 1e-8) / (weit.sum(dim=(2, 3)) + 1e-8)

    # weighted e loss function
    pred = torch.sigmoid(pred)
    mpred = pred.mean(dim=(2, 3)).view(pred.shape[0], pred.shape[1], 1, 1).repeat(1, 1, pred.shape[2], pred.shape[3])
    phiFM = pred - mpred

    mmask = mask.mean(dim=(2, 3)).view(mask.shape[0], mask.shape[1], 1, 1).repeat(1, 1, mask.shape[2], mask.shape[3])
    phiGT = mask - mmask

    EFM = (2.0 * phiFM * phiGT + 1e-8) / (phiFM * phiFM + phiGT * phiGT + 1e-8)
    QFM = (1 + EFM) * (1 + EFM) / 4.0
    eloss = 1.0 - QFM.mean(dim=(2, 3))

    # weighted iou loss function
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1.0 - (inter + 1 + 1e-8) / (union - inter + 1 + 1e-8)

    return (wbce + eloss + wiou).mean()

def train(train_loader, model, optimizer, epoch, save_path, step_counter):
    model.train()
    loss_all = 0
    epoch_step = 0
    size_rates = [0.75, 1]  # removed 1.25 to fit 8GB VRAM
    loss_g_record, loss_c_record = AvgMeter(), AvgMeter()
    try:
        pbar = tqdm(enumerate(train_loader, start=1), total=len(train_loader),
                     desc=f'Epoch {epoch:03d}/{opt.epoch:03d}',
                     ncols=100, ascii=False, file=open(os.devnull, "w"))
        for i, (images, gts) in pbar:
            for rate in size_rates:
                optimizer.zero_grad()

                # ---- data prepare ----
                images = images.cuda()
                gts = gts.cuda()

                # ---- rescale ----
                trainsize = int(round(opt.trainsize * rate / 32) * 32)
                if rate != 1:
                    images = F.interpolate(images, size=(trainsize, trainsize), mode='bicubic', align_corners=True)
                    gts = F.interpolate(gts, size=(trainsize, trainsize), mode='bicubic', align_corners=True)

                # ---- forward ----
                edge, cam_out_2, cam_out_3, cam_out_4 = model(images)

                # ---- loss function ----
                loss_g = hybrid_e_loss(edge, gts)
                loss_c = hybrid_e_loss(cam_out_2, gts) + hybrid_e_loss(cam_out_3, gts) + hybrid_e_loss(cam_out_4, gts)
                loss = loss_g + loss_c

                # ---- backward ----
                loss.backward()
                clip_gradient(optimizer, opt.clip)
                optimizer.step()
                step_counter['val'] += 1
                epoch_step += 1
                loss_all += loss.data

                # ---- recording loss ----
                if rate == 1:
                    loss_g_record.update(loss_g.data, opt.batchsize)
                    loss_c_record.update(loss_c.data, opt.batchsize)

            # ---- train visualization ----
            pbar.set_postfix({'loss_g': f'{loss_g_record.show():.4f}', 'loss_c': f'{loss_c_record.show():.4f}'})

            if i % 10 == 0 or i == total_step:
                logging.info(
                    '[Train Info]:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], loss_g: {:0.4f}, loss_c: {:0.4f}]'.
                        format(epoch, opt.epoch, i, total_step, loss_g_record.show(), loss_c_record.show()))

        loss_all /= epoch_step
        logging.info('[Train Info]: Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
        if epoch % 10 == 0:
            torch.save(model.state_dict(), save_path + 'Net_epoch_{}.pth'.format(epoch))
            print('[Saving Snapshot:]', save_path + 'Net_epoch_{}.pth'.format(epoch))
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path + 'Net_epoch_{}.pth'.format(epoch))
        print('Save checkpoints successfully!')
        raise


def test(test_loader, model, epoch, save_path, best_state):
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        for i in range(test_loader.size):
            _, image, gt, _, _ = test_loader.load_data()
            gt = np.asarray(gt, np.float32) / 255.0  # fix: normalize to [0,1]
            image = image.cuda()
            edge, cam_out_2, cam_out_3, cam_out_4 = model(image)

            res = F.interpolate(cam_out_4, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])
        mae = mae_sum / test_loader.size

        if epoch == 1:
            best_state['mae'] = mae
            print('[CAMO Cur Epoch: {}] Metrics MAE={}'.format(epoch, mae))
        else:
            if mae < best_state['mae']:
                best_state['mae'] = mae
                best_state['epoch'] = epoch
                torch.save(model.state_dict(), save_path + 'KKK_best_{}.pth'.format(best_state['epoch']))
                print('>>> save state_dict successfully! best epoch is {}.'.format(epoch))
            else:
                print('>>> not find the best epoch -> continue training ...')
            print(
                '[CAMO Cur Epoch: {}] Metrics MAE={}    [CAMO Best Epoch: {}] Metrics MAE={}'.format(
                    epoch, mae, best_state['epoch'], best_state['mae']))
            logging.info(
                '[CAMO Cur Epoch: {}] Metrics MAE={}    CAMO Best Epoch: {}] Metrics MAE={}'.format(
                    epoch, mae, best_state['epoch'], best_state['mae']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=50, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='choosing optimizer AdamW or SGD')
    parser.add_argument('--batchsize', type=int, default=8, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
    parser.add_argument('--model', type=str, default='PVTv2-B2',
                        choices=['PVTv2-B1', 'PVTv2-B2', 'PVTv2-B3', 'PVTv2-B4', 'PVTv2-B5'])
    parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')

    parser.add_argument('--load', type=str, default=None, help='train from checkpoints')

    parser.add_argument('--train_root', type=str,
                        default='./Dataset/TrainValDataset/',
                        help='path to train dataset')
    parser.add_argument('--test_root', type=str,
                        default='./Dataset/TestDataset/CAMO/',
                        help='the test rgb images root')

    parser.add_argument('--save_epoch', type=int, default=5,
                        help='every N epochs save your trained snapshot')
    parser.add_argument('--save_root', type=str,
                         default='./model_pth/OUR/')

    opt = parser.parse_args()

    # set the device for training
    if opt.gpu_id == '0':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print('USE GPU 0')
    elif opt.gpu_id == '1':
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        print('USE GPU 1')
    cudnn.benchmark = True

    # build the model
    model = Network(channel=64).cuda()

    optimizer = torch.optim.AdamW(model.parameters(), opt.lr)

    # ---- resume from checkpoint ----
    start_epoch = 1
    if opt.load:
        import re
        ckpt = torch.load(opt.load, map_location="cuda")
        model.load_state_dict(ckpt, strict=False)
        m = re.search(r"(?:epoch_|KKK_best_)(\d+)", opt.load)
        if m:
            start_epoch = int(m.group(1)) + 1
        print(">>> Resumed from {} continue at epoch {}".format(opt.load, start_epoch))


    save_path = opt.save_root
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # load data
    print('load data...')
    train_loader = get_loader(
        image_root=opt.train_root + 'Imgs/',
        gt_root=opt.train_root + 'GT/',
        batchsize=opt.batchsize,
        trainsize=opt.trainsize,
        num_workers=4
    )
    val_loader = test_dataset(image_root=opt.test_root + 'Imgs/',
                              gt_root=opt.test_root + 'GT/',
                              testsize=opt.trainsize)
    total_step = len(train_loader)

    # logging
    logging.basicConfig(filename=save_path + 'log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info(">>> current mode: network-train/val")
    logging.info('>>> config: {}'.format(opt))
    print('>>> config: : {}'.format(opt))

    step_counter = {'val': 0}
    best_state = {'mae': 1, 'epoch': 0}

    cosine_schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=20, eta_min=1e-5)
    print("Start Training...")

    for epoch in range(start_epoch, opt.epoch):
        # schedule
        cosine_schedule.step()
        logging.info('>>> current lr: {}'.format(cosine_schedule.get_last_lr()[0]))

        # train
        train(train_loader, model, optimizer, epoch, save_path, step_counter)
        test(val_loader, model, epoch, save_path, best_state)