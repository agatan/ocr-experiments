import torch
import torch.nn as nn
import torch.nn.functional as F

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
        x = self.layers(x)
        x = x.squeeze(2)  # squeeze compressed height dimension.
        x = torch.transpose(x, 1, 2)
        return x


class RecognitionLoss(nn.Module):
    def forward(self, recognitions, masks, targets, target_lengths):
        log_probs = F.log_softmax(recognitions, dim=2)  # batch_size * max_box, length, vocab
        log_probs = torch.transpose(log_probs, 0, 1)  # length, batch_size * max_box, vocab
        input_lengths = masks.sum(dim=1)
        return F.ctc_loss(log_probs, targets, input_lengths, target_lengths)


class TrainingModel(nn.Module):
    def __init__(self, backbone, detection, recognition):
        super(TrainingModel, self).__init__()
        self.backbone = backbone
        self.detection = detection
        self.recognition = recognition
        self.recognition_loss = RecognitionLoss()
        self.height = 8

    def forward(self, images, boxes, targets, target_lengths):
        feature_map = self.backbone(images)
        detection = self.detection(feature_map)
        pooled, mask = roirotate(feature_map, boxes, height=self.height)
        batch_size, max_box, channel, height, width = pooled.size()
        pooled = pooled.view(batch_size * max_box, channel, height, width)
        mask = mask.view(batch_size * max_box, width)
        recognitions = self.recognition(pooled)
        targets = targets.view(batch_size * max_box, -1)
        target_lengths = target_lengths.view(batch_size * max_box)
        recognition_loss = self.recognition_loss(recognitions, mask, targets, target_lengths)
        return detection, recognition_loss
