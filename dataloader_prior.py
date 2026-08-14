"""dataloader_prior.py - Data loader with LLM priors (pos_map + tag_bow).

Each sample returns: (image, gt, pos_map, tag_bow)
  image  (3,H,W) RGB normalized
  gt     (1,H,W) binary mask [0,1]
  pos_map (1,H,W) bbox gaussian heatmap prior
  tag_bow (V,)    semantic bag-of-words

Data augmentation keeps prior alignment for flip; soft prior tolerates
small crop/rotate offsets by design.
"""
import os
import random
import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms


def cv_random_flip(img, label, pos_map):
    if random.randint(0, 1) == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        if pos_map is not None:
            pos_map = pos_map.transpose(Image.FLIP_LEFT_RIGHT)
    return img, label, pos_map


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


class CamObjDatasetPrior(data.Dataset):
    def __init__(self, image_root, gt_root, prior_root, trainsize, vocab_size):
        self.trainsize = trainsize
        self.vocab_size = vocab_size
        self.images = sorted([image_root + f for f in os.listdir(image_root)
                              if f.endswith(".jpg") or f.endswith(".png")])
        self.gts = sorted([gt_root + f for f in os.listdir(gt_root)
                           if f.endswith(".jpg") or f.endswith(".png")])
        self.prior_root = prior_root

        # filter matching pairs
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
        """img_name: 'camourflage_00001.jpg' -> prior npz path"""
        npz_name = f"train_{img_name.replace('.', '_')}.npz"
        npz_path = os.path.join(self.prior_root, npz_name)
        if os.path.exists(npz_path):
            data = np.load(npz_path)
            pos_map = data["pos_map"]          # (1, 352, 352)
            tag_bow = data["tag_bow"]          # (V,)
        else:
            pos_map = np.zeros((1, 352, 352), dtype=np.float32)
            tag_bow = np.zeros(self.vocab_size, dtype=np.float32)
        return pos_map, tag_bow

    def __getitem__(self, index):
        img_path = self.images[index]
        gt_path = self.gts[index]
        img_name = os.path.basename(img_path)

        image = Image.open(img_path).convert("RGB")
        gt = Image.open(gt_path).convert("L")

        # augmentation
        image, gt = randomCrop(image, gt)
        image, gt = randomRotation(image, gt)
        image, gt, _ = cv_random_flip(image, gt, None)  # flip only img+gt

        # load priors (pos_map stays 352x352, no geometric aug)
        pos_map, tag_bow = self._load_prior(img_name)

        # resize + tensor
        image_t = self.img_transform(image)
        gt_t = self.gt_transform(gt)
        pos_t = torch.from_numpy(pos_map).float()
        if pos_t.shape[-1] != self.trainsize:
            pos_t = torch.nn.functional.interpolate(
                pos_t.unsqueeze(0), size=(self.trainsize, self.trainsize),
                mode="bilinear", align_corners=False).squeeze(0)
        tag_t = torch.from_numpy(tag_bow).float()

        return image_t, gt_t, pos_t, tag_t

    def __len__(self):
        return self.size


def get_prior_loader(image_root, gt_root, prior_root, batchsize, trainsize,
                     vocab_size, shuffle=True, num_workers=4, pin_memory=True):
    dataset = CamObjDatasetPrior(image_root, gt_root, prior_root, trainsize, vocab_size)
    loader = data.DataLoader(dataset, batch_size=batchsize, shuffle=shuffle,
                             num_workers=num_workers, pin_memory=pin_memory)
    return loader