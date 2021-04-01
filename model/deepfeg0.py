import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

class ASPP_module(nn.Module):
    def __init__(self, inplanes, planes, dilation):
        super(ASPP_module, self).__init__()
        if dilation == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = dilation
        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.GroupNorm(16, planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class DeepLabv3_plus(nn.Module):
    def __init__(self, nInputChannels=3, n_classes=2, os=8, pretrained=False):
        super(DeepLabv3_plus, self).__init__()

        # Atrous Conv
        self.resnet_features = ResNet50(nInputChannels, os, pretrained=pretrained)

        for p in self.resnet_features.parameters():
            p.requires_grad = False
        # for i, p in enumerate(self.resnet_features.parameters()):
        #     # 102 layer3 的后三个模块参与训练
        #     if i < 102:
        #         p.requires_grad = False
        # ASPP
        dilations = [1, 4, 8, 16]
        self.aspp1 = ASPP_module(1024, 64, dilation=dilations[0])
        self.aspp2 = ASPP_module(1024, 64, dilation=dilations[1])
        self.aspp3 = ASPP_module(1024, 64, dilation=dilations[2])
        self.aspp4 = ASPP_module(1024, 64, dilation=dilations[3])
        self.relu = nn.ReLU()
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(1024, 64, 1, stride=1, bias=False),
                                             nn.GroupNorm(4, 64),
                                             nn.ReLU())
        self.conv1 = nn.Conv2d(320, 64, 1, bias=False)
        self.bn1 = nn.GroupNorm(8, 64)

        self.conv2 = nn.Conv2d(256, 32, 1, bias=False)
        self.bn2 = nn.GroupNorm(2, 32)

        self.last_conv = nn.Sequential(nn.Conv2d(96, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.GroupNorm(4, 64),
                                       nn.ReLU(),
                                       nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.GroupNorm(4, 64),
                                       nn.ReLU(),
                                       nn.Conv2d(64, 1, kernel_size=1, stride=1))

    def forward(self, input):

        x, low_level_features = self.resnet_features(input)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.upsample(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = F.upsample(x, size=(int(math.ceil(input.size()[-2]/4)),
                                int(math.ceil(input.size()[-1]/4))), mode='bilinear', align_corners=True)

        low_level_features = self.conv2(low_level_features)
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)


        x = torch.cat((x, low_level_features), dim=1)
        x = self.last_conv(x)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()


class Attn_mudule(nn.Module):
    def __init__(self, level, out_ch):
        super().__init__()
        self.level = level
        # s8,s4,s2
        inter_dim = [128, 128, 128]
        # s4
        c = 16
        if self.level == 's8':
            self.inter_dim = inter_dim[0]
            self.resize_s8 = add_conv(1024, inter_dim[0], 1, 1)
            self.resize_s4 = add_conv(256, inter_dim[0], 3, 2)
            self.resize_s2 = add_conv(64, inter_dim[0], 3, 2)
            self.expand = add_conv(inter_dim[0], out_ch, 3, 1)

        if self.level == 's4':
            self.inter_dim = inter_dim[1]
            self.resize_s4 = add_conv(256, inter_dim[1], 1, 1)
            self.resize_s8 = add_conv(1024, inter_dim[1], 1, 1)
            self.resize_s2 = add_conv(64, inter_dim[1], 3, 2)
            self.expand = add_conv(inter_dim[1], out_ch, 3, 1)
        if self.level == 's2':
            self.inter_dim = inter_dim[2]
            self.resize_s2 = add_conv(64, inter_dim[2], 3, 1)
            self.resize_s4 = add_conv(256, inter_dim[2], 1, 1)
            self.resize_s8 = add_conv(1024, inter_dim[2], 1, 1)
            self.expand = add_conv(inter_dim[2], out_ch, 3, 1)

        self.weight_s4 = add_conv(self.inter_dim, 16, 1, 1)
        self.weight_s8 = add_conv(self.inter_dim, 16, 1, 1)
        self.weight_s2 = add_conv(self.inter_dim, 16, 1, 1)
        self.weight_levels = nn.Conv2d(c * 3, 3, kernel_size=1, stride=1, padding=0)

    def forward(self, s8, s4, s2):

        if self.level == 's8':
            s2 = F.max_pool2d(s2, 3, stride=2, padding=1)
            s2 = self.resize_s2(s2)
            s4 = self.resize_s4(s4)
            s8 = self.resize_s8(s8)
        if self.level == 's4':
            s2 = self.resize_s2(s2)
            s4 = self.resize_s4(s4)
            s8 = self.resize_s8(s8)
            s8 = F.upsample(s8, size=s4.size()[2:], mode='bilinear', align_corners=True)
        if self.level == 's2':
            s2 = self.resize_s2(s2)
            s4 = self.resize_s4(s4)
            s8 = self.resize_s8(s8)
            s4 = F.upsample(s4, size=s2.size()[2:], mode='bilinear')
            s8 = F.upsample(s8, size=s2.size()[2:], mode='bilinear')

        weight_s2 = self.weight_s2(s2)
        weight_s4 = self.weight_s4(s4)
        weight_s8 = self.weight_s8(s8)
        levels = self.weight_levels(torch.cat([weight_s2, weight_s4, weight_s8], 1))
        levels_weight = F.softmax(levels, dim=1)

        fused_out_reduced = s2 * levels_weight[:, 0:1, :, :] + \
                            s4 * levels_weight[:, 1:2, :, :] + \
                            s8 * levels_weight[:, 2:, :, :]
        return self.expand(fused_out_reduced)


def add_conv(in_ch, out_ch, ksize, stride, num_groups=1, leaky=True):
    """
    Add a conv2d / batchnorm / leaky ReLU block.
    Args:
        in_ch (int): number of input channels of the convolution layer.
        out_ch (int): number of output channels of the convolution layer.
        ksize (int): kernel size of the convolution layer.
        stride (int): stride of the convolution layer.
    Returns:
        stage (Sequential) : Sequential layers composing a convolution block.
    """
    stage = nn.Sequential()
    pad = (ksize - 1) // 2
    stage.add_module('conv', nn.Conv2d(in_channels=in_ch,
                                       out_channels=out_ch, kernel_size=ksize, stride=stride,
                                       padding=pad, bias=False))
    stage.add_module('group_norm', nn.GroupNorm(num_groups, out_ch))
    if leaky:
        stage.add_module('leaky', nn.LeakyReLU(0.1))
    else:
        stage.add_module('relu', nn.ReLU(inplace=True))
    return stage


def get_1x_lr_params(model):
    """
    This generator returns all the parameters of the net except for
    the last classification layer. Note that for each batchnorm layer,
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
    any batchnorm parameter
    """
    b = [model.resnet_features]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k


def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last layer of the net,
    which does the classification of pixel into classes
    """
    b = [model.aspp1, model.aspp2, model.aspp3, model.aspp4, model.conv1, model.attn_s2, model.attn_s4,
         model.attn_s8, model.conv_s2, model.conv_s4, model.conv_s8, model.pred_s2, model.pred_s4, model.pred_s8]
    for j in range(len(b)):
        for k in b[j].parameters():
            if k.requires_grad:
                yield k


if __name__ == "__main__":
    model = DeepLabv3_plus(nInputChannels=3, n_classes=1, os=16, pretrained=True)
    model.eval()
    image = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        output = model.forward(image)

    import torch

    # model = ResNet50(pretrained=True, os=16)

    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('total params:{}'.format(total_num))
    print('trainable params:{}'.format(trainable_num))
