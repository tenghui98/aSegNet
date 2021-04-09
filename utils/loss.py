import torch
import torch.nn as nn
from torch import Tensor
# from .dice_loss import DiceLoss
import torch.nn.functional as F
import numpy as np
from focal_loss.focal_loss import FocalLoss

class SegmentationLosses(object):
    def __init__(self, weight, size_average=True, ignore_index=2, batch_average=True, cuda=True):
        self.weight = weight
        self.ignore_index = ignore_index
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        elif mode == 'l1ce':
            return self.L1CELoss
        else:
            raise NotImplementedError

    def L1CELoss(self, logit, target):

        s1 = torch.squeeze(logit, 1)
        mask = torch.ne(target, self.ignore_index)
        target = target[mask]
        s1 = s1[mask]
        # s1_loss = F.binary_cross_entropy_with_logits(s1, target, reduction='mean', pos_weight=self.weight)
        s1_loss = self.FocalLoss(s1, target, gamma=2, alpha=0.75)
        # s16, s4, s1 = logit['s16'], logit['s4'], logit['s1']
        # n, c, h, w = s1.size()
        # pred_s1 = torch.sigmoid(s1)
        # s1 = torch.squeeze(s1, 1)
        # s4 = torch.squeeze(s4, 1)
        # s16 = torch.squeeze(s16, 1)
        # pred_s1 = torch.squeeze(pred_s1, 1)
        # mask = torch.ne(target, self.ignore_index)
        # target = target[mask]
        # s1 = s1[mask]
        # s4 = s4[mask]
        # s16 = s16[mask]
        # pred_s1 = pred_s1[mask]
        #
        # # s1_loss = F.binary_cross_entropy_with_logits(s1, target, reduction='mean', pos_weight=self.weight)
        # s1_loss = 2 * (1 - F.cosine_similarity(pred_s1, target, dim=0))
        # s4_loss = F.binary_cross_entropy_with_logits(s4, target, reduction='mean', pos_weight=self.weight)
        # s16_loss = F.binary_cross_entropy_with_logits(s16, target, reduction='mean', pos_weight=self.weight)
        #
        loss = s1_loss

        return loss

    def CrossEntropyLoss(self, logit, target):

        n, c, h, w = logit.size()
        weights = torch.tensor([0.25,0.75])
        criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

    def FocalLoss(self, logit, target, gamma=1, alpha=0.5):
        # with torch.no_grad():
        #     alphas = torch.empty_like(target).fill_(1 - alpha)
        #     alphas[target == 1] = alpha
        criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_index,reduction='mean')
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** gamma) * logpt

        return loss




if __name__ == "__main__":
    loss = SegmentationLosses(cuda=True)
    torch.manual_seed(1)
    a = torch.rand(5, 2, 240, 320).cuda()
    b = torch.rand(5, 240, 320).cuda()
    print(loss.CrossEntropyLoss(a, b).item())
    # print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    # print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())
