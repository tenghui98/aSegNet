import os
from tqdm import tqdm
import numpy as np
from mypath import Path
from sklearn.utils import compute_class_weight


def calculate_weigths_labels(args, dataloader, num_classes):
    # Create an instance from the data loader
    z = np.zeros((num_classes,))
    # Initialize tqdm
    tqdm_batch = tqdm(dataloader)
    print('Calculating classes weights')
    for sample in tqdm_batch:
        y = sample['label']
        y = y.detach().cpu().numpy()
        mask = (y == 0) | (y == 1)
        labels = y[mask].astype(np.uint8)
        count_l = np.bincount(labels, minlength=num_classes)
        z += count_l
    tqdm_batch.close()
    total_frequency = np.sum(z)
    class_weights = []
    for frequency in z:
        class_weight = 1 / (np.log(1.02 + (frequency / total_frequency)))
        class_weights.append(class_weight)

    # ret = np.array(class_weights[1] / class_weights[0])
    ret = np.array(class_weights)
    # classes_weights_path = os.path.join(Path.root_dir('img'), args.category, args.scene, args.scene + '_classes_weights.npy')
    # np.save(classes_weights_path, ret)

    return ret
