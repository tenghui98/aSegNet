import torch
import random
import numpy as np

from PIL import Image, ImageOps, ImageFilter
import matplotlib.pyplot as plt


class Normalize(object):

    def __init__(self, args, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std
        self.args = args

    def __call__(self, sample):
        img = sample['image']
        gt = sample['label']
        img = np.array(img).astype(np.float32)
        gt = np.array(gt).astype(np.float32)
        # not use in FegNet
        img /= 255.0
        img -= self.mean
        img /= self.std
        # hard shadow 50 , outside region of interest(ROI) 85,

        gt[np.where(gt == 50)] = 0
        gt[np.where((gt == 170) | (gt == 85))] = 2
        gt[np.where(gt == 255)] = 1

        # val_gt = np.floor(val_gt)
        return {'image': img,
                'label': gt}


class Normlize_test(object):
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        img_tmp = np.array(img).astype(np.float32)
        # not use in FegNet
        img_tmp /= 255.0
        img_tmp -= self.mean
        img_tmp /= self.std
        img_tmp = np.array(img_tmp).astype(np.float32).transpose((2, 0, 1))
        img = torch.from_numpy(img_tmp).float()

        return img


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        mask = sample['label']

        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.float32)

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()

        return {'image': img,
                'label': mask}


class RandomGaussianBlur(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))

        return {'image': img,
                'label': mask}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        return {'image': img,
                'label': mask}


class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 1), int(self.base_size * 1.5))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'label': mask}


def decode_segmap(label_mask, plot=False):
    label_colours = np.asarray([[0, 0, 0], [255, 255, 255]])
    n_classes = 2
    # VOID_LABEL = 2
    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]

    r[label_mask == 2] = 170
    g[label_mask == 2] = 170
    b[label_mask == 2] = 170

    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb


def decode_segmap_sequence(label_masks):
    masks = []
    for label_mask in label_masks:
        masks.append(decode_segmap(label_mask))
    masks = torch.from_numpy(np.array(masks).transpose([0, 3, 1, 2]))
    return masks
