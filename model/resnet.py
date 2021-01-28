import math
from collections import OrderedDict

import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from model.cbam import CBAM

def load_weights_sequential(target, source_state):
    new_dict = OrderedDict()

    for k1, v1 in target.state_dict().items():
        if k1 in source_state:
            tar_v = source_state[k1]

            if v1.shape != tar_v.shape:
                # Init the new segmentation channel with zeros
                # print(v1.shape, tar_v.shape)
                c, _, w, h = v1.shape
                tar_v = torch.cat([
                    tar_v,
                    torch.zeros((c, 3, w, h)),
                ], 1)

            new_dict[k1] = tar_v

    target.load_state_dict(new_dict)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, rate=1, downsample=None, use_cbam=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               dilation=rate, padding=rate, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.rate = rate
        if use_cbam:
            self.cbam = CBAM(planes * 4, 16)
        else:
            self.cbam = None

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

        if not self.cbam is None:
            out = self.cbam(out)
        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, nInputChannels, block, layers, os=16, pretrained = True,use_cbam=False):
        self.inplanes = 64
        super(ResNet, self).__init__()
        if os == 16:
            strides = [1, 2, 2, 1]
            rates = [1, 1, 1, 2]
            blocks = [1, 2, 4]
        elif os == 8:
            strides = [1, 2, 1, 1]
            rates = [1, 1, 2, 2]
            blocks = [1, 2, 1]
        else:
            raise NotImplementedError

        # Modules
        self.conv1 = nn.Conv2d(nInputChannels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], use_cbam, stride=strides[0], rate=rates[0])
        self.layer2 = self._make_layer(block, 128, layers[1], use_cbam, stride=strides[1], rate=rates[1])
        self.layer3 = self._make_layer(block, 256, layers[2], use_cbam, stride=strides[2], rate=rates[2])
        # self.layer4 = self._make_layer(block, 512, layers[3], use_cbam, stride=strides[3], rate=rates[3])
        self.layer4 = self._make_MG_unit(block, 512, use_cbam,blocks=blocks, stride=strides[3], rate=rates[3])

        self._init_weight()
        if pretrained:
            self._load_pretrained_model()

    def _make_layer(self, block, planes, blocks, use_cbam, stride=1, rate=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, rate, downsample, use_cbam))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, use_cbam, blocks=[1, 2, 4], stride=1, rate=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, rate=blocks[0] * rate, downsample=downsample, use_cbam=use_cbam))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1, rate=blocks[i] * rate))

        return nn.Sequential(*layers)

    def _load_pretrained_model(self):
        pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth')
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        low_level_feat = x
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)
        return x,  low_level_feat

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def ResNet50(nInputChannels=3, os=16, pretrained=True, use_cbam=False):
    model = ResNet(nInputChannels, Bottleneck, [3, 4, 6, 3], os, pretrained=pretrained,use_cbam=use_cbam)
    return model

if __name__ == "__main__":
    import torch
    model = ResNet50(pretrained=True, os=16)
    input = torch.rand(10, 3, 512, 512)
    output, f_1,f_2 = model(input)
    print(output.size())
    print(f_1.size())
    print(f_2.size())

    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('total params:{}'.format(total_num))
    print('trainable params:{}'.format(trainable_num))