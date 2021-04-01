from .cdw2014_train import CDW_Train
from .cdw2014_test import CDW_Test
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import numpy as np
from mypath import Path
import os


def make_data_loader( args, **kwargs):

    category, scene = args.category,args.scene
    gt_dir = os.path.join(Path.root_dir('gt'), category, scene + str(200))
    img_dir = os.path.join(Path.root_dir('img'), category, scene, 'input')

    train_and_val = CDW_Train(args,gt_dir=gt_dir, img_dir=img_dir)
    n_classes = train_and_val.NUM_CLASSES

    indices = list(range(len(train_and_val)))
    np.random.seed(42)
    np.random.shuffle(indices)
    split = int(np.floor(len(train_and_val) * args.train_rate))
    train_indices = indices[:split]
    val_indices = indices[split:]
    train_set = Subset(train_and_val, indices=train_indices)
    val_set = Subset(train_and_val, indices=val_indices)

    test_set = CDW_Test(img_dir = img_dir)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(test_set,batch_size=args.test_batch_size,shuffle=False,**kwargs)
    return train_loader,val_loader,test_loader,n_classes
