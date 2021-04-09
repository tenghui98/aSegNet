import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    def __init__(self, in_ch=128, dilation=1):
        super().__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_ch, in_ch // 2, kernel_size=1),
                                   nn.GroupNorm(in_ch // 32, in_ch // 2),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_ch // 2, in_ch // 2, kernel_size=3, padding=dilation, dilation=dilation),
            nn.GroupNorm(4, in_ch // 2),
            nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(in_ch // 2, in_ch, kernel_size=1),
                                   nn.GroupNorm(in_ch // 16, in_ch),
                                   nn.ReLU())
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out += identity
        return self.relu(out)


class Encoder(nn.Module):
    def __init__(self, in_ch=1024, encoder_ch=128, dilations=None):
        super(Encoder, self).__init__()
        if dilations is None:
            dilations = [1, 2, 4, 8]
        self.conv0 = nn.Conv2d(in_ch, encoder_ch, kernel_size=1)
        self.gn0 = nn.GroupNorm(encoder_ch // 16, encoder_ch)
        # self.gap = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
        #                          nn.Conv2d(in_ch, encoder_ch, 1, stride=1, bias=False),
        #                          nn.GroupNorm(encoder_ch//16, encoder_ch),
        #                          nn.ReLU())
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Sequential(
            nn.Conv2d(encoder_ch, encoder_ch, kernel_size=3, padding=dilations[0], dilation=dilations[0]),
            nn.GroupNorm(encoder_ch // 16, encoder_ch),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(encoder_ch, encoder_ch, kernel_size=3, padding=dilations[1], dilation=dilations[1]),
            nn.GroupNorm(encoder_ch // 16, encoder_ch),
            nn.ReLU())
        self.conv3 = nn.Sequential(
            nn.Conv2d(encoder_ch, encoder_ch, kernel_size=3, padding=dilations[2], dilation=dilations[2]),
            nn.GroupNorm(encoder_ch // 16, encoder_ch),
            nn.ReLU())
        self.conv4 = nn.Sequential(
            nn.Conv2d(encoder_ch, encoder_ch, kernel_size=3, padding=dilations[3], dilation=dilations[3]),
            nn.GroupNorm(encoder_ch // 16, encoder_ch),
            nn.ReLU())

        self.last_conv = nn.Sequential(nn.Conv2d(encoder_ch * 5, encoder_ch, kernel_size=1, bias=False),
                                       nn.GroupNorm(encoder_ch // 16, encoder_ch),
                                       nn.ReLU())
        self.dropout = nn.Dropout(0.5)


    def forward(self, features):
        out = self.gn0(self.conv0(features))
        d1 = self.conv1(out)
        d2 = self.conv2(d1)
        d3 = self.conv3(d2)
        d4 = self.conv4(d3)

        gap = self.gap(out)
        gap = F.upsample(gap, size=out.size()[2:], mode='bilinear', align_corners=True)

        out = self.last_conv(torch.cat([d1, d2, d3, d4, gap], 1))

        return self.dropout(out)


class CAB(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(CAB, self).__init__()
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.sigmod = nn.Sigmoid()

    def forward(self, x1, x2):
        # high, low
        x = torch.cat([x1, x2], dim=1)
        x = self.global_pooling(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.sigmod(x)
        x2 = x * x2
        res = x2 + x1
        return res


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        out = x * y
        return out


if __name__ == '__main__':
    f = torch.rand(1, 1024, 16, 16)
    encoder = Encoder()
    out = encoder(f)
    print(out.size())
    total_num = sum(p.numel() for p in encoder.parameters())
    trainable_num = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print('total params:{}'.format(total_num))
    print('trainable params:{}'.format(trainable_num))
