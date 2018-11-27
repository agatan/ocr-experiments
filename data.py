import random
import os
import json
from typing import List

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image


class CharDictionary:
    def __init__(self, chars):
        self._char2idx = {"<PAD>": 0, "<UNK>": 1}
        self._idx2char = ["<PAD>", "<UNK>"]
        for c in chars:
            idx = len(self._idx2char)
            self._char2idx[c] = idx
            self._idx2char.append(c)
        self.pad_value = 0
        self.unknown_value = 1

    def char2idx(self, c):
        return self._char2idx.get(c, self.unknown_value)

    def idx2char(self, i):
        if i < len(self._idx2char):
            return self._idx2char[i]
        return "<UNK>"

    @property
    def vocab(self):
        return len(self._idx2char)


def _has_file_allowed_extension(filename: str, extensions: List[str]):
    filename = filename.lower()
    return any(filename.endswith(ext) for ext in extensions)


class Dataset(data.Dataset):
    def __init__(
            self,
            root,
            chardict,
            image_size,
            feature_map_scale,
            transform=None,
            image_extensions=["jpg", "jpeg", "png"],
    ):
        super().__init__()
        self.root = os.path.expanduser(root)
        self.chardict = chardict
        self.image_size = image_size
        self.feature_map_scale = feature_map_scale
        self.feature_map_size = (
            self.image_size[0] // feature_map_scale, self.image_size[1] // feature_map_scale)
        self.transform = transform
        self._gather_files(image_extensions)

    def _gather_files(self, image_extensions):
        candidates = [os.path.join(self.root, f) for f in os.listdir(
            self.root) if _has_file_allowed_extension(f, image_extensions)]
        samples = []
        for candidate in candidates:
            index = candidate.rfind(".")
            if index < 0:
                continue
            annot = candidate[:index] + ".json"
            if os.path.exists(annot):
                samples.append((candidate, annot))
        self.samples = samples

    def _compute_ground_truth_box(self, boxes):
        ground_truth = torch.zeros((5,) + self.feature_map_size)
        boxes = boxes.float()
        xmin = boxes[:, 0] / self.feature_map_scale
        zeros = torch.zeros_like(xmin)
        xmin_ceil = torch.max(xmin.ceil(), zeros).long()
        xmin_floor = torch.max(xmin.floor(), zeros).long()
        xmax = boxes[:, 2] / self.feature_map_scale
        xmax_limits = torch.ones_like(xmax) * self.feature_map_size[1]
        xmax_ceil = torch.min(xmax.ceil(), xmax_limits).long()
        xmax_floor = torch.min(xmax.floor(), xmax_limits).long()
        ymin = boxes[:, 1] / self.feature_map_scale
        ymin_ceil = torch.max(ymin.ceil(), zeros).long()
        ymin_floor = torch.max(ymin.floor(), zeros).long()
        ymax = boxes[:, 3] / self.feature_map_scale
        ymax_limits = torch.ones_like(ymax) * self.feature_map_size[0]
        ymax_ceil = torch.min(ymax.ceil(), ymax_limits).long()
        ymax_floor = torch.min(ymax.floor(), ymax_limits).long()
        for i in range(boxes.size(0)):
            ground_truth[0, ymin_floor[i]:ymin_ceil[i],
                         xmin_floor[i]:xmin_ceil[i]] = -1
            ground_truth[0, ymax_floor[i]:ymax_ceil[i],
                         xmax_floor[i]:xmax_ceil[i]] = -1
            ground_truth[0, ymin_ceil[i]:ymax_floor[i],
                         xmin_ceil[i]:xmax_floor[i]] = 1
            ground_truth[1, ymin_ceil[i]:ymax_floor[i], xmin_ceil[i]:xmax_floor[i]] = torch.arange(xmin_ceil[i], xmax_floor[i]) - xmin[i]
            ground_truth[2, ymin_ceil[i]:ymax_floor[i], xmin_ceil[i]:xmax_floor[i]] = torch.arange(ymin_ceil[i], ymax_floor[i]).unsqueeze(1) - ymin[i]
            ground_truth[3, ymin_ceil[i]:ymax_floor[i], xmin_ceil[i]:xmax_floor[i]] = xmax[i] - torch.arange(xmin_ceil[i], xmax_floor[i])
            ground_truth[4, ymin_ceil[i]:ymax_floor[i], xmin_ceil[i]:xmax_floor[i]] = ymax[i] - torch.arange(ymin_ceil[i], ymax_floor[i]).unsqueeze(1)
        return ground_truth

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        imagefile, annotfile = self.samples[index]
        with open(imagefile, "rb") as f:
            image = Image.open(f).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        with open(annotfile, "r") as f:
            annot = json.load(f)
            boxes = annot['boxes']
            random.shuffle(boxes)
            annot['boxes'] = boxes[:1]
        length_of_longest_text = max(
            (len(box['text']) for box in annot['boxes']))
        padded_texts = torch.zeros(
            (len(annot['boxes']), length_of_longest_text), dtype=torch.int32)
        padded_texts[...] = self.chardict.pad_value
        target_lengths = torch.zeros((len(annot['boxes']),), dtype=torch.long)
        for i, box in enumerate(annot['boxes']):
            text = [self.chardict.char2idx(c) for c in box['text']]
            padded_texts[i, :len(text)] = torch.tensor(text)
            target_lengths[i] = len(text)
        boxes = torch.zeros((len(annot['boxes']), 4), dtype=torch.int32)
        for i, box in enumerate(annot['boxes']):
            boxes[i, 0] = box['left']
            boxes[i, 1] = box['top']
            boxes[i, 2] = box['left'] + box['width']
            boxes[i, 3] = box['top'] + box['height']
        return image, boxes, self._compute_ground_truth_box(boxes), padded_texts, target_lengths

    def collate_fn(self, data):
        batch_size = len(data)
        images, boxes, ground_truths, texts, target_lengths = zip(*data)
        images = torch.stack(images, dim=0)
        max_box = max((len(bs) for bs in boxes))
        padded_boxes = torch.zeros((batch_size, max_box, 4), dtype=torch.int32)
        max_text_length = max((text.size()[-1] for text in texts))
        padded_texts = torch.zeros((batch_size, max_box, max_text_length), dtype=torch.long)
        padded_texts[...] = self.chardict.pad_value
        padded_target_lengths = torch.zeros((batch_size, max_box), dtype=torch.long)
        padded_ground_truths = torch.zeros((batch_size, 5) + self.feature_map_size)
        for i, bs in enumerate(boxes):
            padded_boxes[i, :len(bs), :] = bs
        for i, ts in enumerate(texts):
            padded_texts[i, :len(ts), :ts.size()[-1]] = ts
            padded_target_lengths[i, :len(ts)] = target_lengths[i]
        for i, gt in enumerate(ground_truths):
            padded_ground_truths[i, :, :, :] = gt
        return images, padded_boxes, padded_ground_truths, padded_texts, padded_target_lengths


def reconstruct_boxes(boxes_pred: torch.Tensor):
    """
    Args:
        boxes_pred: [4 (L, T, R, B), #boxes]
    """
    recons = torch.zeros_like(boxes_pred)
    _, h, w = boxes_pred.size()
    xx = torch.arange(0, w).float()
    yy = torch.arange(0, h).float().unsqueeze(1).repeat(1, w)
    recons[0, :, :] = xx - boxes_pred[0, :, :]
    recons[1, :, :] = yy - boxes_pred[1, :, :]
    recons[2, :, :] = xx + boxes_pred[2, :, :]
    recons[3, :, :] = yy + boxes_pred[3, :, :]
    return recons
