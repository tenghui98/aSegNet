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

        roi_path = os.path.join(Path.root_dir('img'), args.category, args.scene, 'ROI.bmp')
        ROI = Image.open(roi_path).convert('L')
        ROI = np.array(ROI).astype(np.float32)
        idx = np.where(ROI == 0.)

        s16, s4, s1 = logit['s16'], logit['s4'], logit['s1']

        s4 = torch.sigmoid(s4)
        s4 = (torch.squeeze(s4) > 0.7).detach().cpu().numpy().astype('int')
        s4[idx] = 0.
        s4 = torch.from_numpy(s4).unsqueeze(0)

        s16 = torch.sigmoid(s16)
        s16 = (torch.squeeze(s16) > 0.7).detach().cpu().numpy().astype('int')
        s16[idx] = 0.
        s16 = torch.from_numpy(s16).unsqueeze(0)

        s1 = torch.sigmoid(s1)
        s1 = (torch.squeeze(s1) > 0.7).detach().cpu().numpy().astype('int')
        s1[idx] = 0.
        s1 = torch.from_numpy(s1).unsqueeze(0)

        grid_image = make_grid([s1, s4, s16], 3, padding=5, pad_value=1)
        writer.add_image('Predicted label', grid_image, global_step)

        grid_image = make_grid(decode_segmap_sequence(torch.squeeze(target[:1], 1).detach().cpu().numpy()),
                               3, normalize=False, range=(0, 255))
        writer.add_image('Groundtruth label', grid_image, global_step)