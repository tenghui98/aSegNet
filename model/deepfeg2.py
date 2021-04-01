import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo


def gn_init(m, zero_init=False):
    assert isinstance(m, nn.GroupNorm)
    m.weight.data.fill_(0. if zero_init else 1.)
    m.bias.data.zero_()


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.GroupNorm(32, planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               dilation=dilation, padding=dilation, bias=False)
        self.bn2 = nn.GroupNorm(32, planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.GroupNorm(32, 4 * planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

        gn_init(self.bn1)
        gn_init(self.bn2)
        gn_init(self.bn3, zero_init=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, nInputChannels, block, layers, os=16, pretrained=False):
        self.inplanes = 64
        super(ResNet, self).__init__()

        strides = [1, 2, 1, 1]
        dilations = [1, 1, 2, 4]
        blocks = [1, 2, 4]
        # Modules
        self.conv1 = nn.Conv2d(nInputChannels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.GroupNorm(32, 64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], dilation=dilations[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2])
        # self.layer4 = self._make_MG_unit(block, 512, blocks=blocks, stride=strides[3], dilation=dilations[3])

        self._init_weight()
        gn_init(self.bn1)

        if pretrained:
            self._load_pretrained_model()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(32, planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks=[1, 2, 4], stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(32, planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=blocks[0] * dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1, dilation=blocks[i] * dilation))

        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.conv1(input)
        os2 = x
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        os4 = x
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)
        return x, os4, os2

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def _load_pretrained_model(self):
        pretrain_dict = model_zoo.load_url(
            url='https://github.com/ppwwyyxx/GroupNorm-reproduce/releases/download/v0.1/ImageNet-ResNet50-GN.pth',
            file_name='ImageNet-ResNet50-GN.pth')
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict['state_dict'].items():
            kk = k[7:]
            if kk in state_dict:
                model_dict[kk] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)


def ResNet50(nInputChannels=3, os=8, pretrained=False):
    model = ResNet(nInputChannels, Bottleneck, [3, 4, 6, 3], os, pretrained=pretrained)
    return model


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

        for p in self.parameters():
            p.requires_grad = False
        # ASPP
        dilations = [1, 4, 8, 16]
        self.aspp1 = ASPP_module(1024, 256, dilation=dilations[0])
        self.aspp2 = ASPP_module(1024, 256, dilation=dilations[1])
        self.aspp3 = ASPP_module(1024, 256, dilation=dilations[2])
        self.aspp4 = ASPP_module(1024, 256, dilation=dilations[3])
        self.relu = nn.ReLU()
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(1024, 256, 1, stride=1, bias=False),
                                             nn.GroupNorm(32, 256),
                                             nn.ReLU())
        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.GroupNorm(32, 256)
        self.attn_s2 = Attn_mudule('s2', out_ch=32)
        self.attn_s4 = Attn_mudule('s4', out_ch=32)
        self.attn_s8 = Attn_mudule('s8', out_ch=32)

        self.conv_s8 = add_conv(256 + 32, 256, 3, 1, 32)
        self.conv_s4 = add_conv(256 + 32 + 1, 256, 3, 1, 32)
        self.conv_s2 = add_conv(256 + 32 + 1, 256, 3, 1, 32)
        self.pred_s8 = nn.Conv2d(256, 1, kernel_size=1, stride=1)
        self.pred_s4 = nn.Conv2d(256, 1, kernel_size=1, stride=1)
        self.pred_s2 = nn.Conv2d(256, 1, kernel_size=1, stride=1)

    def forward(self, input):
        x, os4, os2 = self.resnet_features(input)
        # x = self.res_dropout(res_feat_os16)
        os8 = x
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.upsample(x5, size=x1.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # s8
        feat_s8 = self.attn_s8(os8, os4, os2)
        x = self.conv_s8(torch.cat([x, feat_s8], dim=1))
        out_s8 = self.pred_s8(x)
        up_s8 = F.upsample(out_s8, size=(int(math.ceil(input.size()[-2] / 4)),
                                         int(math.ceil(input.size()[-1] / 4))), mode='bilinear', align_corners=True)
        x = F.upsample(x, size=(int(math.ceil(input.size()[-2] / 4)),
                                int(math.ceil(input.size()[-1] / 4))), mode='bilinear', align_corners=True)

        # s4
        feat_os4 = self.attn_s4(os8, os4, os2)
        x = self.conv_s4(torch.cat([x, feat_os4, up_s8], dim=1))
        out_s4 = self.pred_s4(x)
        up_s4 = F.upsample(out_s4, size=(int(math.ceil(input.size()[-2] / 2)),
                                         int(math.ceil(input.size()[-1] / 2))), mode='bilinear', align_corners=True)
        x = F.upsample(x, size=(int(math.ceil(input.size()[-2] / 2)),
                                int(math.ceil(input.size()[-1] / 2))), mode='bilinear', align_corners=True)
        # s2
        feat_os2 = self.attn_s2(os8, os4, os2)
        x = self.conv_s2(torch.cat([x, feat_os2, up_s4], dim=1))
        out_s2 = self.pred_s2(x)

        s1 = F.upsample(out_s2, size=input.size()[2:], mode='bilinear', align_corners=True)
        s4 = F.upsample(out_s4, size=input.size()[2:], mode='bilinear', align_corners=True)
        s8 = F.upsample(out_s8, size=input.size()[2:], mode='bilinear', align_corners=True)

        return {'s1': s1,
                's4': s4,
                's8': s8}

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
        inter_dim = [64, 64, 64]
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
            # self.resize_s2 = add_conv(64, inter_dim[2], 3, 1)
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
            # s2 = self.resize_s2(s2)
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


class AASPP(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.weight1 = add_conv(in_ch, 8, 1, 1)
        self.weight2 = add_conv(in_ch, 8, 1, 1)
        self.weight3 = add_conv(in_ch, 8, 1, 1)
        self.weight4 = add_conv(in_ch, 8, 1, 1)
        self.weight5 = add_conv(in_ch, 8, 1, 1)

        self.weight_levels = nn.Conv2d(8 * 5, 5, kernel_size=1, stride=1, padding=0)
        self.expand = add_conv(in_ch, out_ch, 3, 1, num_groups=16)

    def forward(self, x1, x2, x3, x4, x5):
        w1 = self.weight1(x1)
        w2 = self.weight1(x2)
        w3 = self.weight1(x3)
        w4 = self.weight1(x4)
        w5 = self.weight1(x5)
        levels = self.weight_levels(torch.cat([w1, w2, w3, w4, w5], 1))
        levels_weight = F.softmax(levels, dim=1)
        fused_out_reduced = x1 * levels_weight[:, 0:1, :, :] + \
                            x2 * levels_weight[:, 1:2, :, :] + \
                            x3 * levels_weight[:, 2:3, :, :] + \
                            x4 * levels_weight[:, 3:4, :, :] + \
                            x4 * levels_weight[:, 4:, :, :]
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