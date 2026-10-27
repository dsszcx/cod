import os
import random
import numpy as np
from PIL import Image, ImageEnhance
import cv2

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from utils import seed_worker



def cv_random_flip(img, label):
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
    return img, label


def randomCrop(image, label):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), label.crop(random_region)


def randomRotation(image, label):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
    return image, label


def colorEnhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image


def randomGaussianNoise(image, mean=0.1, sigma=0.35):
    """添加高斯噪声（兼容 RGB 和灰度图）"""
    img = np.asarray(image, dtype=np.float32)
    noise = np.random.normal(mean, sigma, img.shape)
    img = img + noise
    img = np.clip(img, 0, 255)
    return Image.fromarray(np.uint8(img))


def randomPepperNoise(img, noise_ratio=0.0015):
    """添加椒盐噪声（兼容 RGB 和灰度图）"""
    img = np.array(img)
    h, w = img.shape[:2]
    n_channels = img.shape[2] if img.ndim == 3 else 1
    noise_num = int(noise_ratio * h * w)
    for _ in range(noise_num):
        x = random.randint(0, h - 1)
        y = random.randint(0, w - 1)
        if n_channels == 1:
            img[x, y] = 0 if random.randint(0, 1) == 0 else 255
        else:
            val = 0 if random.randint(0, 1) == 0 else 255
            img[x, y, :] = val
    return Image.fromarray(img)


class CamObjDataset(data.Dataset):
    def __init__(self, image_root, gt_root, trainsize):
        self.trainsize = trainsize
        # get filenames
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')
                       or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]

        # sorted files
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)

        # filter mathcing degrees of files
        self.filter_files()
        # transforms
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        # get size of dataset
        self.size = len(self.images)
        print('>>> trainig/validing with {} samples'.format(self.size))

    def __getitem__(self, index):
        # read assest/gts/edges/depths
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])

        # data augumentation
        image, gt = cv_random_flip(image, gt)
        image, gt = randomCrop(image, gt)
        image, gt = randomRotation(image, gt)

        image = colorEnhance(image)
        gt = randomPepperNoise(gt)

        image = self.img_transform(image)
        gt = self.gt_transform(gt)

        return image, gt

    def filter_files(self):
        assert len(self.images) == len(self.gts) and len(self.gts) == len(self.images)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size


# dataloader for training
def get_loader(image_root, gt_root, batchsize, trainsize,
               shuffle=True, num_workers=12, pin_memory=True, seed=0):
    dataset = CamObjDataset(image_root, gt_root, trainsize)
    # seeded generator: without it the batch order of two --seed-identical runs
    # differed, because it came from whatever the global RNG state happened to be
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory,
                                  generator=gen,
                                  worker_init_fn=seed_worker)
    return data_loader



'''test dataset and loader'''
class test_dataset:
    """Test-time dataset.  Images and GT are paired by file STEM.

    The old code zipped two independently sorted lists, so a single missing GT
    shifted *every* subsequent pair: image ``00012`` was scored against GT
    ``00018`` and nothing complained.  Pairing by stem makes that impossible;
    any unpaired file is counted and printed instead.
    """

    IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')
    GT_EXTS = ('.tif', '.tiff', '.png', '.jpg', '.bmp')

    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        image_root = image_root if image_root.endswith(("/", "\\")) else image_root + "/"
        gt_root = gt_root if gt_root.endswith(("/", "\\")) else gt_root + "/"
        self.image_root, self.gt_root = image_root, gt_root

        images = sorted(f for f in os.listdir(image_root)
                        if f.lower().endswith(self.IMG_EXTS))
        gts = sorted(f for f in os.listdir(gt_root)
                     if f.lower().endswith(self.GT_EXTS))
        gt_by_stem = {os.path.splitext(g)[0]: g for g in gts}

        self.images, self.gts, self.missing_gt = [], [], []
        for f in images:
            stem = os.path.splitext(f)[0]
            if stem in gt_by_stem:
                self.images.append(image_root + f)
                self.gts.append(gt_root + gt_by_stem[stem])
            else:
                self.missing_gt.append(f)
        self.n_images_on_disk = len(images)
        self.n_gt_on_disk = len(gts)
        if self.missing_gt:
            print(f">>> [test_dataset] WARNING: {len(self.missing_gt)} image(s) have "
                  f"no GT and are NOT evaluated (first: {self.missing_gt[0]})")
        assert self.images, (f"no image/GT pair: {image_root} has {len(images)} "
                             f"image(s), {gt_root} has {len(gts)} GT(s)")

        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def reset(self):
        self.index = 0

    def load_data(self):
        image_o = self.rgb_loader(self.images[self.index])
        image = self.transform(image_o).unsqueeze(0)
        image_o = cv2.imread(self.images[self.index])
        image_o = self.gt_transform(image_o)

        gt = self.binary_loader(self.gts[self.index])

        # forward slashes on every platform: the caller derives the prior npz
        # name from this string ('\\' would keep the whole path)
        name = self.images[self.index].replace("\\", "/").split("/")[-1]

        image_for_post = self.rgb_loader(self.images[self.index])
        # image_for_post = image_for_post.resize(gt.size)
        image_for_post = self.gt_transform(image_for_post).unsqueeze(0)

        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'

        self.index += 1
        self.index = self.index % self.size

        return image_o, image, gt, name, image_for_post

        # return image, gt, name, image_for_post

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size
