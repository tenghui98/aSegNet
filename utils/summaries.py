import os
import torch
from PIL import Image
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
from dataloaders.custom_transforms import decode_segmap_sequence
import numpy as np
from mypath import Path
from dataloaders.custom_transforms import decode_segmap_sequence

class TensorboardSummary(object):
    def __init__(self, directory):
        self.directory = directory

    def create_summary(self):
        writer = SummaryWriter(log_dir=os.path.join(self.directory))
        return writer

    def visualize_image(self, args, writer, image, target, logit, global_step):

        grid_image = make_grid(image[:1].clone().cpu().data, 3, normalize=True)
        writer.add_image('Image', grid_image, global_step)
        logit = logit['s1']
        # grid_image = make_grid(decode_segmap_sequence(torch.max(logit[:3], 1)[1].detach().cpu().numpy()),
        #                        3, normalize=False, range=(0, 255))
        # writer.add_image('Predicted label', grid_image, global_step)
        logit = torch.sigmoid(logit)
        logit = (torch.squeeze(logit)>0.5).detach().cpu().numpy().astype('int')
        logit = torch.from_numpy(logit).unsqueeze(0)
        writer.add_image('Predicted label', logit, global_step)

        grid_image = make_grid(decode_segmap_sequence(torch.squeeze(target[:1], 1).detach().cpu().numpy()),
                               3, normalize=False, range=(0, 255))
        writer.add_image('Groundtruth label', grid_image, global_step)