import torch
import random
import numpy as np

from PIL import Image, ImageOps, ImageFilter

class Normalize(object):

    def __call__(self,sample):
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        # not use in FegNet
        img /= 255.0

        shape = mask.shape
        mask /= 255.0
        mask = mask.reshape(-1)
        idx = np.where(np.logical_and(mask>0.25,mask<0.8))[0]
        if len(idx) >0:
            mask[idx] = -1.
        mask = mask.reshape(shape)
        mask = np.floor(mask)

        return {'image':img,
                'label':mask}


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