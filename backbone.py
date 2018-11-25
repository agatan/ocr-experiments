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
