import math

import torch
from torch import Tensor
import torch.nn as nn
from torch.utils import model_zoo
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, dilation=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, dilation=dilation,
                               padding=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, dilation=dilation,
                               padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x: Tensor):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class resnet18(nn.Module):
    def __init__(self, layers=None, pretrained=True):
        if layers is None:
            layers = [2, 2, 2, 2]
        self.inplanes = 64
        super(resnet18, self).__init__()
        strides = [1, 2, 1, 1]
        dilations = [1, 1, 2, 2]

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlock, 64, layers[0], stride=strides[0], dilation=dilations[0])
        self.layer2 = self._make_layer(BasicBlock, 128, layers[1], stride=strides[1], dilation=dilations[1])
        self.layer3 = self._make_layer(BasicBlock, 256, layers[2], stride=strides[2], dilation=dilations[2])
        self.layer4 = self._make_layer(BasicBlock, 512, layers[3], stride=strides[3], dilation=dilations[3])
        # self.layer4 = self._make_MG_unit(BasicBlock, 512, blocks=blocks, stride=strides[3], dilation=dilations[3])
        self._init_weight()
        if pretrained:
            self._load_pretrained_model()

    def forward(self, x: Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        low_level_feat = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x, low_level_feat

    def _make_layer(self, basicblock, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * basicblock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * basicblock.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * basicblock.expansion),
            )
        layers = []
        layers.append(basicblock(self.inplanes, planes, stride=stride, dilation=dilation,
                                 downsample=downsample))
        self.inplanes = planes * basicblock.expansion
        for i in range(1, blocks):
            layers.append(basicblock(self.inplanes, planes, stride=1, dilation=dilation))

        return nn.Sequential(*layers)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model(self):
        pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet18-5c106cde.pth')
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)


class ASPP(nn.Module):
    def __init__(self, inplanes, planes, dilation):
        super(ASPP, self).__init__()
        if dilation == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = dilation
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, padding=padding, dilation=dilation,
                                     bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self._init_weights()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class DeepFeg(nn.Module):
    def __init__(self, n_classes=2):
        super(DeepFeg, self).__init__()
        self.resnet = resnet18()
        dialtions = [1, 6, 12, 18]

        self.aspp1 = ASPP(512, 128, dilation=dialtions[0])
        self.aspp2 = ASPP(512, 128, dilation=dialtions[1])
        self.aspp3 = ASPP(512, 128, dilation=dialtions[2])
        self.aspp4 = ASPP(512, 128, dilation=dialtions[3])
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),
                                             nn.Conv2d(512,128,kernel_size=1,bias=False),
                                             nn.ReLU())
        self.conv1 = nn.Conv2d(128*5,128,1,bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 16, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(16)

        self.last_conv = nn.Sequential(nn.Conv2d(128+16, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(64),
                                       nn.ReLU(),
                                       nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(64),
                                       nn.ReLU(),
                                       nn.Conv2d(64, n_classes, kernel_size=1, stride=1))
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.2)
        self._init_weights()

    def forward(self,input):
        x, low_level_features = self.resnet(input)
        x = self.dropout1(x)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.dropout2(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = F.interpolate(x, size=(int(math.ceil(input.size()[-2]/4)),
                                int(math.ceil(input.size()[-1]/4))), mode='bilinear', align_corners=True)

        low_level_features = self.conv2(low_level_features)
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)

        x = torch.cat((x, low_level_features), dim=1)
        x = self.last_conv(x)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def get_1x_lr_params(model):
    """
    This generator returns all the parameters of the net except for
    the last classification layer. Note that for each batchnorm layer,
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
    any batchnorm parameter
    """
    b = [model.resnet]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k


def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last layer of the net,
    which does the classification of pixel into classes
    """
    b = [model.aspp1, model.aspp2, model.aspp3, model.aspp4, model.conv1, model.conv2, model.last_conv]
    for j in range(len(b)):
        for k in b[j].parameters():
            if k.requires_grad:
                yield k


if __name__ == "__main__":
    import torch
    model = DeepFeg(n_classes=2)
    input = torch.rand(10, 3, 512, 512)
    output = model(input)
    print(output.size())
