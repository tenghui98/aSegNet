import os
import torch
from PIL import Image
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
from dataloaders.custom_transforms import decode_segmap_sequence
import numpy as np
from mypath import Path

class TensorboardSummary(object):
    def __init__(self, directory):
        self.directory = directory

    def create_summary(self):
        writer = SummaryWriter(log_dir=os.path.join(self.directory))
        return writer

    def visualize_image(self, args, writer, image, target, logit, global_step):

        writer.add_image('Image', torch.squeeze(image,0), global_step)
        # visual pred
        logit = torch.sigmoid(logit)
        logit = (torch.squeeze(logit)>0.5).detach().cpu().numpy().astype('int')
        roi_path = os.path.join(Path.root_dir('img'),args.category,args.scene,'ROI.bmp')
        ROI = Image.open(roi_path)
        ROI = np.array(ROI).astype(np.float32)
        logit[np.where(ROI == 0.)] = 0.
        logit = torch.from_numpy(logit).unsqueeze(0)
        # grid_image = make_grid(logit,3, normalize=False, range=(0, 255))
        writer.add_image('Predicted label', logit, global_step)

        writer.add_image('Groundtruth label',
                         decode_segmap_sequence(torch.squeeze(target[:3], 1).detach().cpu().numpy()).squeeze(0), global_step)