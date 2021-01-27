import os
import torch
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
from dataloaders.custom_transforms import decode_segmap_sequence

class TensorboardSummary(object):
    def __init__(self, directory):
        self.directory = directory

    def create_summary(self):
        writer = SummaryWriter(log_dir=os.path.join(self.directory))
        return writer

    def visualize_image(self, writer, image, target, outputs, global_step):
        # image *= 255.0
        # image = image.int()
        pred_28 = decode_segmap_sequence((torch.squeeze(outputs['pred_28'],1)>0.5).detach().cpu().numpy().astype('int'))
        pred_56 = decode_segmap_sequence((torch.squeeze(outputs['pred_56'],1)>0.5).detach().cpu().numpy().astype('int'))
        pred_224 = decode_segmap_sequence((torch.squeeze(outputs['pred_224'],1)>0.5).detach().cpu().numpy().astype('int'))
        images = torch.cat([pred_28,pred_56,pred_224])

        grid_image = make_grid(image[:3].clone().cpu().data, 3, normalize=True)
        writer.add_image('Image', grid_image, global_step)
        grid_image = make_grid(images,3, normalize=False, range=(0, 255))
        writer.add_image('Predicted label', grid_image, global_step)
        grid_image = make_grid(decode_segmap_sequence(torch.squeeze(target[:3], 1).detach().cpu().numpy()),
                               3, normalize=False, range=(0, 255))
        writer.add_image('Groundtruth label', grid_image, global_step)