import torch.nn as nn

from backbone import ResNet50Backbone


class Network(nn.Module):
    def __init__(self, vocab, pretrained=False):
        super(Network, self).__init__()
        self.backbone = ResNet50Backbone(pretrained=pretrained)
        self.detection_branch = nn.Conv2d(256, 5, kernel_size=1)
        self.recognition_branch = nn.Conv2d(256, vocab, kernel_size=1)

    def forward(self, x):
        feature_map = self.backbone(x)
        detection = self.detection_branch(feature_map)
        return detection
