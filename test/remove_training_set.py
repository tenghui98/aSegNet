import glob
import numpy as np
from mypath import Path
from dataset_dict import dataset
from tqdm import tqdm
import os

def getFiles(scene_input_path):
    inlist = glob.glob(os.path.join(scene_input_path, '*.png'))
    return np.asarray(inlist)

for category, scene_list in dataset.items():
    for scene in tqdm(scene_list,desc=category):
        gt_path = os.path.join(Path.root_dir('gt'),category,scene+str(200))
        result_path = os.path.join(Path.root_dir('result'),category,scene)
        gt_path_list = getFiles(gt_path)
        result_path_list = getFiles(result_path)

        for i in result_path_list:
            re_name = os.path.basename(i).replace('bin', 'gt')
            for j in gt_path_list:
                gt_name = os.path.basename(j)
                if re_name == gt_name:
                    os.remove(i)
