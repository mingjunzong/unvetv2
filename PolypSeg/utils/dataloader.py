"""息肉分割数据集 / DataLoader

相对原版的改动：
  1. mask 不再只认 .png（原版训练集 mask 只收 .png，CVC-ClinicDB 原始 .tif mask 会被全部丢掉）
  2. 图片和 mask 改成按文件名（stem）配对，而不是靠两个排序后列表的下标对齐
  3. augmentation 判断改成 str(...) == 'True'（原版传 bool True 时 `True == 'True'` 为假，增强被静默关掉）
  4. RandomRotation 的 resample= 参数在新版 torchvision 已移除，改用 interpolation=
  5. 路径统一用 os.path.join，文件名用 os.path.basename（原版用 split('/')，Windows 下会出错）
"""
import os
import random

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
MASK_EXT = ('.png', '.tif', '.tiff', '.jpg', '.jpeg', '.bmp')


def _index_by_stem(folder, exts):
    out = {}
    if not os.path.isdir(folder):
        raise FileNotFoundError('目录不存在: {}'.format(folder))
    for f in os.listdir(folder):
        stem, ext = os.path.splitext(f)
        if ext.lower() in exts:
            out[stem] = os.path.join(folder, f)
    return out


def _make_pairs(image_root, gt_root):
    """按文件名配对，返回 [(img_path, gt_path), ...]"""
    images = _index_by_stem(image_root, IMG_EXT)
    gts = _index_by_stem(gt_root, MASK_EXT)

    common = sorted(set(images.keys()) & set(gts.keys()))
    if len(common) == 0:
        raise RuntimeError(
            '图片和 mask 文件名完全没有交集，无法配对。\n'
            '  images: {}\n  masks : {}'.format(image_root, gt_root))
    if len(common) != len(images) or len(common) != len(gts):
        print('[warn] {} / {}: 有 {} 张图、{} 张 mask，按文件名配对成功 {} 对'.format(
            os.path.basename(image_root.rstrip('/\\')),
            os.path.basename(gt_root.rstrip('/\\')),
            len(images), len(gts), len(common)))
    return [(images[s], gts[s]) for s in common]


class PolypDataset(data.Dataset):
    """dataloader for polyp segmentation tasks"""

    def __init__(self, image_root, gt_root, trainsize, augmentations):
        self.trainsize = trainsize
        self.augmentations = augmentations
        self.pairs = _make_pairs(image_root, gt_root)
        self.size = len(self.pairs)
        print('{} pairs, augmentation={}'.format(self.size, augmentations))

        if str(self.augmentations) == 'True':
            print('Using RandomRotation, RandomFlip')
            self.img_transform = transforms.Compose([
                transforms.RandomRotation(90, interpolation=transforms.InterpolationMode.NEAREST),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
            self.gt_transform = transforms.Compose([
                transforms.RandomRotation(90, interpolation=transforms.InterpolationMode.NEAREST),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor()])
        else:
            print('no augmentation')
            self.img_transform = transforms.Compose([
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
            self.gt_transform = transforms.Compose([
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor()])

    def __getitem__(self, index):
        image = self.rgb_loader(self.pairs[index][0])
        gt = self.binary_loader(self.pairs[index][1])

        seed = np.random.randint(2147483647)  # make a seed with numpy generator
        random.seed(seed)
        torch.manual_seed(seed)
        image = self.img_transform(image)

        random.seed(seed)  # 同一 seed，保证图像和 mask 的几何变换一致
        torch.manual_seed(seed)
        gt = self.gt_transform(gt)
        return image, gt

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


def get_loader(image_root, gt_root, batchsize, trainsize, shuffle=True, num_workers=4,
               pin_memory=True, augmentation=False):
    dataset = PolypDataset(image_root, gt_root, trainsize, augmentation)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory,
                                  drop_last=False)
    return data_loader


class test_dataset:
    """逐张加载测试图（同时返回原图尺寸的 gt，用于上采样回原尺寸算指标）"""

    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.pairs = _make_pairs(image_root, gt_root)
        self.size = len(self.pairs)
        self.index = 0
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()

    def load_data(self):
        img_path, gt_path = self.pairs[self.index]
        image = self.rgb_loader(img_path)
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(gt_path)
        name = os.path.basename(img_path)
        if name.lower().endswith('.jpg') or name.lower().endswith('.jpeg'):
            name = os.path.splitext(name)[0] + '.png'
        self.index += 1
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
