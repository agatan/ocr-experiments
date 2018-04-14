import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class Bottleneck(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, 4 * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(4 * planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != 4 * planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, 4 * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(4 * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out


class FPN(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(64, blocks[0], stride=1)
        self.layer2 = self._make_layer(128, blocks[1], stride=2)
        self.layer3 = self._make_layer(256, blocks[2], stride=2)
        self.layer4 = self._make_layer(512, blocks[3], stride=2)
        self.conv6 = nn.Conv2d(4 * 512, 256, kernel_size=3, stride=2, padding=1)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)

        self.latlayer1 = nn.Conv2d(4 * 512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(2 * 512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)

        self.toplayer1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.toplayer2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

    def _make_layer(self, planes, block, stride):
        strides = [stride] + [1] * (block - 1)
        layers = []
        for stride in strides:
            layers.append(Bottleneck(self.in_planes, planes, stride))
            self.in_planes = planes * 4
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def forward(self, x):
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        p6 = self.conv6(c5)
        p7 = self.conv7(F.relu(p6))
        p5 = self.latlayer1(c5)
        p4 = self._upsample_add(p5, self.latlayer2(c4))
        p4 = self.toplayer1(p4)
        p3 = self._upsample_add(p4, self.latlayer3(c3))
        p3 = self.toplayer2(p3)
        return p3, p4, p5, p6, p7


def FPN50():
    return FPN([3, 4, 6, 3])


class SSOCR(nn.Module):
    def __init__(self):
        super().__init__()
        self.fpn = FPN50()
        self.loc_head = self._make_head(9 * 4)

    def forward(self, x):
        fms = self.fpn(x)
        loc_preds = []
        for fm in fms:
            loc_pred = self.loc_head(fm)
            loc_pred = loc_pred.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 4)
            loc_preds.append(loc_pred)
        return torch.cat(loc_preds, 1)

    def _make_head(self, out_chans):
        layers = []
        for _ in range(4):
            layers.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(True))
        layers.append(nn.Conv2d(256, out_chans, kernel_size=3, stride=1, padding=1))
        return nn.Sequential(*layers)
