import torch.nn as nn

from backbone import ResNet50Backbone
from roirotate import roirotate


class Detection(nn.Module):
    def __init__(self):
        super(Detection, self).__init__()
        # [confidence, left, top, right, bottom]
        self.conv = nn.Conv2d(256, 5, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class Recognition(nn.Module):
    def __init__(self, vocab):
        super(Recognition, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Conv2d(256, vocab, kernel_size=1),
        )

    def forward(self, x):
        return self.layers(x)


class TrainingModel(nn.Module):
    def __init__(self, backbone, detection, recognition):
        super(TrainingModel, self).__init__()
        self.backbone = backbone
        self.detection = detection
        self.recognition = recognition
        self.height = 8

    def forward(self, images, boxes):
        feature_map = self.backbone(images)
        detection = self.detection(feature_map)
        pooled, mask = roirotate(feature_map, boxes, height=self.height)
        recognitions = []
        for p in pooled:
            recognitions.append(self.recognition(p))
        recognition = torch.stack(recognitions, dim=0)
        return detection, recognition, mask


import torch

images = torch.zeros((1, 3, 224, 256))
boxes = torch.zeros((1, 3, 4))
boxes = torch.Tensor([[
    [0, 0, 10, 10],
    [0, 0, 100, 200],
]])
print(boxes.size())
model = TrainingModel(ResNet50Backbone(), Detection(), Recognition(10))
detection, recognition, mask = model(images, boxes)
print(detection.size())
print(recognition.size())
print(mask.size())
print(mask)
