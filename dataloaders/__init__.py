from .cdw2014_train import CDW_Train
from .cdw2014_test import CDW_Test
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import numpy as np
from mypath import Path
import os

from .custom_transforms import decode_segmap


def make_data_loader(args, **kwargs):
    category, scene = args.category, args.scene
    gt_dir = os.path.join(Path.root_dir('gt'), category, scene + str(200))
    img_dir = os.path.join(Path.root_dir('img'), category, scene, 'input')

    train_and_val = CDW_Train(args, gt_dir=gt_dir, img_dir=img_dir)
    n_classes = train_and_val.NUM_CLASSES

    indices = list(range(len(train_and_val)))
    np.random.seed(args.seed)
    np.random.shuffle(indices)
    split = int(np.floor(len(train_and_val) * args.train_rate))
    train_indices = indices[:split]
    val_indices = indices[split:]
    train_set = Subset(train_and_val, indices=train_indices)
    val_set = Subset(train_and_val, indices=val_indices)

    test_set = CDW_Test(img_dir=img_dir)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    # decode(val_loader)
    return train_loader, val_loader, test_loader, n_classes


def decode(val_loader):
    import matplotlib.pyplot as plt
    # show randomly 3 pairs of image and its groundtruth
    for i, sample in enumerate(val_loader):
        for j in range(sample['image'].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            mask_tmp = np.array(gt[j]).astype(np.uint8)
            # mask_tmp *= 255.0
            # mask_tmp = mask_tmp.astype(np.uint8)
            mask_tmp = decode_segmap(mask_tmp)
            img_tmp = np.transpose(img[j], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(mask_tmp)

        if i == 10:
            break

    plt.show(block=True)
