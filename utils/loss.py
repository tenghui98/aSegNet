import torch
import torch.nn as nn
from torch import Tensor
from .dice_loss import DiceLoss
import torch.nn.functional as F

class SegmentationLosses(object):
    def __init__(self, weight=None, size_average=True, ignore_index=0.5, batch_average=True, cuda=True):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        elif mode == 'dice':
            return self.DiceLoss
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, images):

        gt = images['gt']
        out_28 = images['out_28']
        out_56= images['out_56']
        out_224 = images['out_224']
        pred_28 = images['pred_28']
        pred_56 = images['pred_56']
        pred_224 = images['pred_224']

        n = gt.size(0)
        mask = torch.ne(gt,self.ignore_index)
        gt = gt[mask]
        pred_28 = pred_28[mask]
        pred_56 = pred_56[mask]
        pred_224 = pred_224[mask]
        out_28 = out_28[mask]
        out_56 = out_56[mask]
        out_224 = out_224[mask]

        ce_weights = [1.0,0.5,0.0]
        l1_weights = [0.0,0.25,1.0]
        l2_weights = [0.0,0.25,1.0]
        # ce_weights = [0.0, 1.0, 0.5, 1.0, 1.0, 0.5]
        # l1_weights = [1.0, 0.0, 0.25, 0.0, 0.0, 0.25]
        # l2_weights = [1.0, 0.0, 0.25, 0.0, 0.0, 0.25]
        # grad_weight = 0.5
        ce_loss = [0] * 3
        l1_loss = [0] * 3
        l2_loss = [0] * 3
        loss = 0.0

        ce_loss[0] = F.binary_cross_entropy_with_logits(out_28, (gt>0.5).float())
        ce_loss[1] = F.binary_cross_entropy_with_logits(out_56, (gt>0.5).float())
        ce_loss[2] = F.binary_cross_entropy_with_logits(out_224, (gt>0.5).float())


        l1_loss[0] = F.l1_loss(pred_28, gt)
        l2_loss[0] = F.mse_loss(pred_28, gt)
        l1_loss[1] = F.l1_loss(pred_56, gt)
        l2_loss[1] = F.mse_loss(pred_56, gt)
        l1_loss[2] = F.l1_loss(pred_224, gt)
        l2_loss[2] = F.mse_loss(pred_224, gt)
        # grad_loss = F.l1_loss(images['gt_sobel'], images['pred_sobel'])

        for i in range(3):
            loss += ce_loss[i] * ce_weights[i] + l1_loss[i] * l1_weights[i] + l2_loss[i] * l2_weights[i]
        # loss += grad_loss*grad_weight


        # target.unsqueeze_(1)
        # # criterion = nn.CrossEntropyLoss(weight=self.weight,reduction='mean')
        # # criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index, reduction='mean')
        # loss = F.binary_cross_entropy_with_logits(logit,target,reduction='mean')

        # if self.batch_average:
        #     loss /= n

        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.25):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index, reduction='mean')
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss

    def DiceLoss(self, logit, target):
        criterion = DiceLoss(weight=self.weight,ignore_index=self.ignore_index)
        if self.cuda:
            criterion = criterion.cuda()
        loss = criterion(logit, target)
        return loss


if __name__ == "__main__":
    loss = SegmentationLosses(cuda=True)
    torch.manual_seed(1)
    a = torch.rand(5, 2, 240, 320).cuda()
    b = torch.rand(5, 240, 320).cuda()
    print(loss.CrossEntropyLoss(a, b).item())
    # print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    # print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())
