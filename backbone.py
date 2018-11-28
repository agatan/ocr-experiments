import torch.nn as nn
import torch.nn.functional as F
import torchvision.models


class ResNet50Backbone(nn.Module):
    def __init__(self, pretrained=False):
        super(ResNet50Backbone, self).__init__()
        self.resnet50 = torchvision.models.resnet50(pretrained=pretrained)
        self.deconv1 = nn.Conv2d(
            2048, 1024, kernel_size=1, stride=1, padding=0)
        self.deconv2 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0)
        self.deconv3 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.resnet50.relu(self.resnet50.bn1(self.resnet50.conv1(x)))
        x = self.resnet50.maxpool(x)
        x1 = self.resnet50.layer1(x)
        x2 = self.resnet50.layer2(x1)
        x3 = self.resnet50.layer3(x2)
        x4 = self.resnet50.layer4(x3)
        x = F.interpolate(self.deconv1(x4), scale_factor=2,
                          mode='bilinear') + x3
        x = F.interpolate(self.deconv2(x), scale_factor=2,
                          mode='bilinear') + x2
        x = F.interpolate(self.deconv3(x), scale_factor=2,
                          mode='bilinear') + x1
        return x


def _conv_bn(in_channels, out_channels, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True),
    )


def _conv1x1_bn(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True),
    )


class _InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(_InvertedResidual, self).__init__()
        self.use_residual = stride == 1 and in_channels == out_channels
        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # depthwise
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, bias=False, groups=in_channels),
                nn.BatchNorm2d(in_channels),
                nn.ReLU6(inplace=True),
                # pointwise
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            hidden_dim = round(in_channels * expand_ratio)
            self.conv = nn.Sequential(
                # pointwise
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # depthwise
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, bias=False, groups=hidden_dim),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pointwise
                nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2Backbone(nn.Module):
    def __init__(self):
        super(MobileNetV2Backbone, self).__init__()
        first_channels = 32
        inverted_residual_config = [
            # t (expand ratio), channels, n (layers), stride
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        self.conv1 = _conv_bn(3, first_channels, stride=2)
        input_channels = first_channels
        self.blocks = nn.ModuleList()
        for i, (t, c, n, s) in enumerate(inverted_residual_config):
            output_channels = c
            layers = []
            for j in range(n):
                if j == 0:
                    layers.append(_InvertedResidual(input_channels, output_channels, stride=s, expand_ratio=t))
                else:
                    layers.append(_InvertedResidual(input_channels, output_channels, stride=1, expand_ratio=t))
                input_channels = output_channels
            self.blocks.append(nn.Sequential(*layers))
        last_channels = 1280
        self.conv2 = _conv1x1_bn(input_channels, last_channels)
        self.deconv1 = nn.Conv2d()

    def forward(self, x):
        x = self.conv1(x)
        for block in self.blocks:
            x = block(x)
        return self.conv2(x)
