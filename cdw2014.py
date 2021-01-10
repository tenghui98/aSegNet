from __future__ import print_function, division

import glob
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from mypath import Path
from torchvision import transforms
import custom_transforms as tr

gt_dir = os.path.join('.', 'cdw2014_train', 'baseline', 'highway200')
img_dir = os.path.join('.', 'cdw2014_dataset', 'baseline', 'highway')


class CDW(Dataset):

    # VOID_LABEL = -1

    def __init__(self, gt_dir=gt_dir, dataset_dir=img_dir, split='train'):
        super().__init__()
        self._gt_dir = gt_dir

        # Search the corresponding images according to the gt list.
        img_dir_list = glob.glob(os.path.join(dataset_dir, 'input', '*.jpg'))
        mask_dir_list = glob.glob(os.path.join(gt_dir, '*.png'))
        temp = []
        for i in range(len(mask_dir_list)):
            mask_name = os.path.basename(mask_dir_list[i])
            mask_idx = mask_name.split('.')[0].split('gt')[1]
            for j in range(len(img_dir_list)):
                img_name = os.path.basename(img_dir_list[j])
                img_idx = img_name.split('.')[0].split('in')[1]
                if mask_idx == img_idx:
                    temp.append(img_dir_list[j])
                    break
        img_dir_list = temp
        assert len(img_dir_list) == len(mask_dir_list)

        self.img_dir_list = sorted(img_dir_list)
        self.mask_dir_list = sorted(mask_dir_list)

    def __len__(self):
        return len(self.img_dir_list)

    def __getitem__(self, item):
        _img, _target = self._make_img_gt_point_pair(item)
        sample = {'image': _img, 'label': _target}
        return self._transform(sample)

    def _make_img_gt_point_pair(self, item):
        _img = Image.open(self.img_dir_list[item]).convert('RGB')
        _target = Image.open(self.mask_dir_list[item])
        return _img, _target

    def _transform(self, sample):
        composed_transforms = transforms.Compose([
            tr.Normalize(),
            tr.ToTensor()
        ])
        return composed_transforms(sample)

    def __str__(self):
        return 'CDW dataset'


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

    cdw_train = CDW()
    dataloader = DataLoader(cdw_train, batch_size=5, shuffle=True)

    # show randomly 5 pairs of image and its groundtruth
    for i, sample in enumerate(dataloader):
        for j in range(sample['image'].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            mask_tmp = np.array(gt[j])
            mask_tmp *= 255.0
            mask_tmp = mask_tmp.astype(np.uint8)

            img_tmp = np.transpose(img[j], axes=[1, 2, 0])
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(mask_tmp, cmap='gray')

        if i == 0:
            break

    plt.show(block=True)
