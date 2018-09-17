import torch
from torch import nn
import torch.nn.functional as F

from ocr.models import mobilenet


class OCRNet(nn.Module):
    def __init__(self):
        super(OCRNet, self).__init__()
        self.backbone = mobilenet.MobileNetV2Backbone()
        self.bbox_conv = nn.Conv2d(self.backbone.last_channel, 5, kernel_size=1, stride=1, padding=0)

        def conv_bn_2(inp, out):
            return nn.Sequential(
                nn.Conv2d(inp, out, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out),
                nn.ReLU6(inplace=True),
                nn.Conv2d(inp, out, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out),
                nn.ReLU6(inplace=True),
            )

        self.ocr_horizontal = nn.Sequential(
            conv_bn_2(self.backbone.last_channel, 64),
            nn.MaxPool2d(kernel_size=(2, 1)),
            conv_bn_2(64, 128),
            nn.MaxPool2d(kernel_size=(2, 1)),
            conv_bn_2(128, 256),
            nn.MaxPool2d(kernel_size=(2, 1)),
        )
        self.ocr_vertical = nn.Sequential(
            conv_bn_2(self.backbone.last_channel, 64),
            nn.MaxPool2d(kernel_size=(1, 2)),
            conv_bn_2(64, 128),
            nn.MaxPool2d(kernel_size=(1, 2)),
            conv_bn_2(128, 256),
            nn.MaxPool2d(kernel_size=(1, 2)),
        )

    def forward(self, x):
        features = self.backbone(x)
        bbox = self.bbox_conv(features)
        return bbox

    def loss_confidence(self, y_pred, y_true):
        alpha = 0.25
        gamma = 2.
        y_true = y_true[:, 0, ...]
        y_pred = y_pred[:, 0, ...]
        y_pred = F.sigmoid(y_pred)
        pt1 = (y_true == 1.0).float() * y_pred + (y_true == 0.).float() * 1.
        pt0 = (y_true == 0.).float() * y_pred + (y_true == 1.).float() * 0.
        pt1_loss = alpha * (1. - pt1).pow(gamma) * torch.log(pt1 + 1e-5)
        pt2_loss = (1 - alpha) * pt0.pow(gamma) * torch.log(1 - pt0 + 1e-5)
        return torch.sum(pt1_loss) + torch.sum(pt2_loss) / y_pred.size()[0]

    def loss_iou(self, y_pred, y_true, eps=1e-5):
        mask = (y_true[:, 0:1, ...] == 1.0).float()
        y_true = y_true * mask
        y_pred = y_pred * mask
        area_true = (y_true[:, 3, ...] - y_true[:, 1, ...]) * (y_true[:, 4, ...] - y_true[:, 2, ...])
        area_pred = (y_pred[:, 3, ...] - y_pred[:, 1, ...]) * (y_pred[:, 4, ...] - y_pred[:, 2, ...])
        x_intersect = torch.max(
            torch.min(y_true[:, 3, ...], y_pred[:, 3, ...]) + torch.min(y_true[:, 1, ...], y_pred[:, 1, ...]),
            torch.zeros(1),
        )
        y_intersect = torch.max(
            torch.min(y_true[:, 4, ...], y_pred[:, 4, ...]) + torch.min(y_true[:, 2, ...], y_pred[:, 2, ...]),
            torch.zeros(1),
        )
        area_intersect = x_intersect * y_intersect
        ious = area_intersect / (area_true + area_pred - area_intersect + eps)
        return - torch.mean(torch.log(ious + eps))