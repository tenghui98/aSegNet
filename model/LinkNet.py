import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.resnet18 import ResNet18
from model.resnet import ResNet50


def get_1x_lr_params(model):
    b = []
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k


def get_10x_lr_params(model):
    b = [model.resnet, model.aspp, model.decoder3, model.decoder2, model.decoder1, model.last_conv]
    # b = [model.aspp, model.conv1, model.last_conv]
    for j in range(len(b)):
        for k in b[j].parameters():
            if k.requires_grad:
                yield k


class DeepFeg(nn.Module):
    def __init__(self, nInputChannels=3, pretrained=False):
        super(DeepFeg, self).__init__()
        # [3,4,3,0]
        self.resnet = ResNet18(nInputChannels, pretrained)
        for i, p in enumerate(self.resnet.parameters()):
            # 102 layer3 的后三个模块参与训练
            if i < 30:
                p.requires_grad = False
        self.aspp = EncoderBlock(in_ch=512, encoder_ch=64)

        # self.conv1 = nn.Conv2d(64, 32, 1, bias=False)
        # self.bn1 = nn.GroupNorm(2,32)
        # self.relu = nn.ReLU()
        # self.last_conv = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #                                nn.GroupNorm(8,128),
        #                                nn.ReLU(),
        #                                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #                                nn.GroupNorm(8,128),
        #                                nn.ReLU(),
        #                                nn.Conv2d(128, 1, kernel_size=1, stride=1))

        self.decoder3 = DecoderBlock(in_ch=64, out_ch=64)
        self.decoder2 = DecoderBlock(in_ch=64, out_ch=64)
        self.decoder1 = DecoderBlock(in_ch=64, out_ch=64)

        self.last_conv = nn.Sequential(nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.GroupNorm(2, 32),
                                       nn.ReLU(),
                                       nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.GroupNorm(2, 32),
                                       nn.ReLU(),
                                       nn.Conv2d(32, 1, kernel_size=1))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, input):
        # x [512,1/8,1/8] e2[128,1/8,1/8] e1[64,1/4,1/4]
        x, e2, e1 = self.resnet(input)

        e3 = self.aspp(x)
        d3 = self.decoder3(e3) + e1
        d2 = self.decoder2(d3)
        d1 = self.decoder1(d2)
        out = self.last_conv(d1)

        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, upsample_scale=None):
        super(DecoderBlock, self).__init__()
        self.scale = upsample_scale
        self.conv1 = nn.Conv2d(in_ch, in_ch // 2, 1, bias=False)
        self.norm1 = nn.GroupNorm(in_ch // 16, in_ch // 2)
        self.relu1 = nn.ReLU()

        self.deconv = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.GroupNorm(in_ch // 16, in_ch // 2)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_ch // 2, out_ch, 1, bias=False)
        self.norm3 = nn.GroupNorm(out_ch // 16, out_ch)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        identity = x
        x = self.relu1(self.norm1(self.conv1(x)))
        x = self.relu2(self.norm2(self.deconv(x)))
        out = self.relu3(self.norm3(self.conv3(x)))
        identity = F.upsample(identity, size=x.size()[2:], mode='bilinear', align_corners=True)
        out += identity
        return out


class EncoderBlock(nn.Module):
    def __init__(self, in_ch=512, encoder_ch=128, dilations=None):
        super(EncoderBlock, self).__init__()
        if dilations is None:
            dilations = [1, 2, 4, 8, 16]
        self.conv1 = nn.Conv2d(in_ch, encoder_ch, kernel_size=1, bias=False)
        self.gn1 = nn.GroupNorm(8, encoder_ch)

        self.conv3 = nn.Conv2d(encoder_ch, encoder_ch, kernel_size=3, padding=1, bias=False)
        self.gn3 = nn.GroupNorm(8, encoder_ch)
        self.relu = nn.ReLU()

        blocks = []
        for i in range(5):
            blocks.append(DilatedBottleneck(encoder_ch, dilation=dilations[i]))

        self.dilated_encoder = nn.Sequential(*blocks)

    def forward(self, x):

        out = self.relu(self.gn1(self.conv1(x)))
        out = self.relu(self.gn3(self.conv3(out)))

        return self.dilated_encoder(out)


class DilatedBottleneck(nn.Module):
    def __init__(self, in_ch=128, dilation=1):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_ch, in_ch // 2, kernel_size=1, padding=0, bias=False),
                                   nn.GroupNorm(8, in_ch // 2),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_ch // 2, in_ch // 2, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            nn.GroupNorm(8, in_ch // 2),
            nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(in_ch // 2, in_ch, kernel_size=1, padding=0, bias=False),
                                   nn.GroupNorm(16, in_ch),
                                   nn.ReLU())

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out + identity
        return out


if __name__ == '__main__':
    input = torch.rand(1, 3, 224, 224)
    model = DeepFeg()
    out = model(input)
    print(out.size())
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('total params:{}'.format(total_num))
    print('trainable params:{}'.format(trainable_num))
