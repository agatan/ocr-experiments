import torch
import torch.nn as nn
import torch.nn.functional as F

from roirotate import roirotate


class Detection(nn.Module):
    def __init__(self):
        super(Detection, self).__init__()
        # [confidence, left, top, right, bottom]
        self.conv = nn.Conv2d(256, 5, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class _BinaryFocalLoss(nn.Module):
    """_BinaryFocalLoss is still buggy...
    """
    def __init__(self, gamma, eps=1e-8, mean=True):
        super(_BinaryFocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.mean = mean

    def forward(self, pred, target):
        p = (target == 0).float() * (1 - pred) + (target == 1).float() * pred
        p = p.clamp(self.eps, 1 - self.eps)
        loss = -(1 - p) ** self.gamma * torch.log(p)
        if self.mean:
            return torch.mean(loss)
        else:
            return torch.sum(loss)


class DetectionLoss(nn.Module):
    def __init__(self, confidence_loss_function='bce'):
        super(DetectionLoss, self).__init__()
        if confidence_loss_function == 'focalloss':
            self.confidence_loss = _BinaryFocalLoss(gamma=2)
        else:
            self.confidence_loss = nn.BCELoss()

    def forward(self, detections, ground_truths):
        """Calculate detector's loss.

        The arguments are consists of 5 channels (confidence, to left, to top, to right, to bottom).

        Args:
            detections (torch.Tensor): output of Detection.
            ground_truths (torch.Tensor): computed ground truth for detection.
        """
        confidences_pred_logits = detections[:, 0, :, :].contiguous().view(-1)
        confidences_gt = ground_truths[:, 0, :, :].contiguous().view(-1)
        care_indices = confidences_gt != -1
        confidences_pred = torch.sigmoid(confidences_pred_logits[care_indices])
        confidences_gt = confidences_gt[care_indices]
        confidences_accuracy = torch.sum((confidences_pred > 0.5).float() == confidences_gt) / torch.ones_like(confidences_gt).float().sum()
        confidences_loss = self.confidence_loss(confidences_pred, confidences_gt)
        return confidences_loss, confidences_accuracy


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
    def __init__(self, backbone, detection, recognition, confidence_loss_function):
        super(TrainingModel, self).__init__()
        self.backbone = backbone
        self.detection = detection
        self.detection_loss = DetectionLoss(confidence_loss_function=confidence_loss_function)
        self.recognition = recognition
        self.recognition_loss = RecognitionLoss()
        self.height = 8

    def forward(self, images, boxes, ground_truths, targets, target_lengths):
        feature_map = self.backbone(images)

        # detection
        detection = self.detection(feature_map)
        detection_loss, confidences_accuracy = self.detection_loss(detection, ground_truths)

        # recognition
        pooled, mask = roirotate(feature_map, boxes, height=self.height)
        batch_size, max_box, channel, height, width = pooled.size()
        pooled = pooled.view(batch_size * max_box, channel, height, width)
        mask = mask.view(batch_size * max_box, width)
        targets = targets.view(batch_size * max_box, -1)
        target_lengths = target_lengths.view(batch_size * max_box)
        non_zero_indices = target_lengths != 0
        pooled = pooled[non_zero_indices, :, :, :]
        mask = mask[non_zero_indices, :]
        targets = targets[non_zero_indices]
        target_lengths = target_lengths[non_zero_indices]
        recognitions = self.recognition(pooled)
        recognition_loss = self.recognition_loss(recognitions, mask, targets, target_lengths)
        return detection_loss, confidences_accuracy, recognition_loss
