""" Full assembly of the parts to form the complete network """

import torch
import torch.nn.functional as F

from model.unet.unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.dropout = nn.Dropout(p=0.2)
        self.up1 = Up(512, 512 // factor, bilinear=bilinear)
        self.up2 = Up(256, 256 // factor, bilinear=bilinear)
        self.up3 = Up(128, 128 // factor, bilinear=bilinear)
        self.up4 = Up(64, 64, bilinear=bilinear)
        self.outc = OutConv(64, n_classes)

        self._init_weights()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.dropout(x)
        x = self.up2(x, x3)
        x = self.dropout(x)
        x = self.up3(x, x2)
        x = self.dropout(x)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()


if __name__ == '__main__':
    input = torch.rand(1, 3, 128, 128)
    model = UNet(n_channels=3, n_classes=1)
    out = model(input)
    print(out.size())
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('total params:{}'.format(total_num))
    print('trainable params:{}'.format(trainable_num))
