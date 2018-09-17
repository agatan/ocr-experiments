import torch
from torch import nn
import torch.nn.functional as F

from ocr.models import mobilenet


class OCRNet(nn.Module):
    def __init__(self, features_pixel=4):
        super(OCRNet, self).__init__()
        self.backbone = mobilenet.MobileNetV2Backbone()
        self.bbox_conv = nn.Conv2d(self.backbone.last_channel, 5, kernel_size=1, stride=1, padding=0)
        self.features_pixel = features_pixel

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
        y_pred = torch.sigmoid(y_pred)
        pt1 = (y_true == 1.0).float() * y_pred + (y_true == 0.).float() * 1.
        pt0 = (y_true == 0.).float() * y_pred + (y_true == 1.).float() * 0.
        pt1_loss = alpha * (1. - pt1).pow(gamma) * torch.log(pt1 + 1e-5)
        pt2_loss = (1 - alpha) * pt0.pow(gamma) * torch.log(1 - pt0 + 1e-5)
        return (-torch.sum(pt1_loss) - torch.sum(pt2_loss)) / (y_pred.size()[0] * y_pred.size()[1] * y_pred.size()[2])

    def loss_iou(self, y_pred, y_true, eps=1e-5):
        y_true = y_true.view(-1, 5)
        y_pred = y_pred.view(-1, 5)
        mask = y_true[:, 0] == 1.0
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        area_true = (y_true[:, 3, ...] + y_true[:, 1, ...]) * (y_true[:, 4, ...] + y_true[:, 2, ...])
        area_pred = torch.max(
            (y_pred[:, 3, ...] + y_pred[:, 1, ...]) * (y_pred[:, 4, ...] + y_pred[:, 2, ...]),
            torch.zeros(1).to(y_pred.device),
        )
        x_intersect = torch.max(
            torch.min(y_true[:, 3, ...], y_pred[:, 3, ...]) + torch.min(y_true[:, 1, ...], y_pred[:, 1, ...]),
            torch.zeros(1).to(y_pred.device),
        )
        y_intersect = torch.max(
            torch.min(y_true[:, 4, ...], y_pred[:, 4, ...]) + torch.min(y_true[:, 2, ...], y_pred[:, 2, ...]),
            torch.zeros(1).to(y_pred.device),
        )
        area_intersect = x_intersect * y_intersect
        ious = area_intersect / (area_true + area_pred - area_intersect + eps)
        return -torch.mean(torch.log(ious + eps))

    def reconstruct_bbox(self, bbox_output):
        bbox_output = bbox_output.cpu()
        batch, _, height, width = bbox_output.size()
        xx = torch.range(0, width - 1).view(1, 1, -1).repeat(batch, height, 1).view(batch, 1, height, width)
        yy = torch.range(0, height - 1).view(1, -1, 1).repeat(batch, 1, width).view(batch, 1, height, width)
        left = xx - bbox_output[:, 1:2, ...]
        right = xx + bbox_output[:, 3:4, ...]
        top = yy - bbox_output[:, 2:3, ...]
        bottom = yy + bbox_output[:, 4:5, ...]
        conf = torch.sigmoid(bbox_output[:, 0, ...])
        boxes = self.features_pixel * torch.cat([left, top, right, bottom], 1)
        conf = conf.view(batch, -1)
        boxes = boxes.view(batch, 4, -1)
        boxes = torch.transpose(boxes, 2, 1)
        return conf, boxes


# Original author: Francisco Massa:
# https://github.com/fmassa/object-detection.torch
# Ported to PyTorch by Max deGroot (02/01/2017)
def nms(boxes, scores, overlap=0.5, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """

    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w*h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep, count