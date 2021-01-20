from __future__ import print_function, division

import glob
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from dataloaders import custom_transforms as tr
from dataloaders.custom_transforms import decode_segmap

class CDW_Test(Dataset):
    # VOID_LABEL = -1
    NUM_CLASSES = 2

    def __init__(self,img_dir):
        super().__init__()
        # Search the corresponding images according to the gt list.
        img_dir_list = glob.glob(os.path.join(img_dir, '*.jpg'))
        self.img_dir_list = img_dir_list


    def __len__(self):
        return len(self.img_dir_list)

    def __getitem__(self, item):
        _img = Image.open(self.img_dir_list[item]).convert('RGB')
        trans = tr.Normlize_test(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        img = trans(_img)
        return img

    def __str__(self):
        return ' All CDW dataset test'


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    from mypath import Path

    cdw_test = CDW_Test(img_dir=os.path.join(Path.root_dir('img'),'baseline','highway','input'))
    dataloader = DataLoader(cdw_test, batch_size=3, shuffle=False)

    for i, img in enumerate(dataloader):
        img_tmp = img[0].numpy()
        img_tmp = np.transpose(img_tmp, axes=[1, 2, 0])
        img_tmp *= 255.0
        img_tmp = img_tmp.astype(np.uint8)
        plt.imshow(img_tmp)
        if i == 0:
            break

    plt.show(block=True)
