import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from model.Modules import Encoder, CAB
from model.cbam import CBAM
from model.resnet18 import ResNet18
from model.resnet import ResNet50
from model.aspp import ASPP


class DeepLabv3_plus(nn.Module):
    def __init__(self, nInputChannels=3, pretrained=False):
        super(DeepLabv3_plus, self).__init__()

        # Atrous Conv
        self.resnet_features = ResNet50(nInputChannels, pretrained=pretrained)
        self.aspp = ASPP(mid_ch=256)

        self.compress = nn.Sequential(nn.Conv2d(256, 48, kernel_size=1, bias=False),
                                      nn.GroupNorm(3, 48),
                                      nn.ReLU())

        self.dropout = nn.Dropout2d(p=0.25)
        self.last_conv = nn.Sequential(nn.Conv2d(256+48, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.GroupNorm(16, 256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.GroupNorm(16, 256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, 1, kernel_size=1))

    def forward(self, input):

        x, os4 = self.resnet_features(input)
        x = self.dropout(x)
        x = self.aspp(x)

        x = F.upsample(x, size=(int(math.ceil(input.size()[-2] / 4)),
                                int(math.ceil(input.size()[-1] / 4))), mode='bilinear', align_corners=True)

        compress_os4 = self.compress(os4)
        x = torch.cat([x, compress_os4], 1)

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


if __name__ == "__main__":
    model = DeepLabv3_plus(nInputChannels=3, pretrained=True)
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
