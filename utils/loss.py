import torch
import torch.nn as nn
from torch import Tensor
# from .dice_loss import DiceLoss
import torch.nn.functional as F
import numpy as np


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
        s1_loss = F.binary_cross_entropy_with_logits(s1, target, reduction='mean', pos_weight=self.weight)
        loss = s1_loss
        return loss

    # def CrossEntropyLoss(self, logit, target):
    #
    #     n, c, h, w = logit.size()
    #     criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
    #                                     size_average=self.size_average)
    #     if self.cuda:
    #         criterion = criterion.cuda()
    #
    #     loss = criterion(logit, target.long())
    #
    #     if self.batch_average:
    #         loss /= n
    #
    #     return loss

    def FocalLoss(self, logit, target, gamma=1, alpha=0.25):
        with torch.no_grad():
            alphas = torch.empty_like(logit).fill_(1 - alpha)
            alphas[target == 1] = alpha

        logpt = -F.binary_cross_entropy_with_logits(logit, target, reduction='none', pos_weight=self.weight)
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** gamma) * alphas * logpt

        loss = loss.mean()

        # if self.batch_average:
        #     loss /= n

        return loss


class DiceLoss(nn.Module):
    def __init__(self, ignore_index):
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, logit, target):
        smooth = 1
        intersection = logit * target
        loss = 2 * (intersection.sum() + smooth) / (logit.sum() + target.sum() + smooth)
        loss = 1 - loss

        return loss


class FocalLossV1(nn.Module):

    def __init__(self, alpha=0.25, gamma=2, reduction='mean', ):
        super(FocalLossV1, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.crit = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, label):
        '''
        logits and label have same shape, and label data type is long
        args:
            logits: tensor of shape (N, ...)
            label: tensor of shape(N, ...)
        Usage is like this:
            >>> criteria = FocalLossV1()
            >>> logits = torch.randn(8, 19, 384, 384)# nchw, float/half
            >>> lbs = torch.randint(0, 19, (8, 384, 384)) # nchw, int64_t
            >>> loss = criteria(logits, lbs)
        '''

        # compute loss
        logits = logits.float()  # use fp32 if logits is fp16
        with torch.no_grad():
            alpha = torch.empty_like(logits).fill_(1 - self.alpha)
            alpha[label == 1] = self.alpha

        probs = torch.sigmoid(logits)
        pt = torch.where(label == 1, probs, 1 - probs)
        ce_loss = self.crit(logits, label.float())
        loss = (alpha * torch.pow(1 - pt, self.gamma) * ce_loss)
        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss


if __name__ == "__main__":
    loss = SegmentationLosses(cuda=True)
    torch.manual_seed(1)
    a = torch.rand(5, 2, 240, 320).cuda()
    b = torch.rand(5, 240, 320).cuda()
    print(loss.CrossEntropyLoss(a, b).item())
    # print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    # print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())
