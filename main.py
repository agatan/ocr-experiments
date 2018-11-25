import os
import json
import math
from typing import List

import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import datagen
from group_norm import GroupNormalization


INPUT_SIZE = np.array([304, 192])
FEATURE_SIZE = INPUT_SIZE // 4


def meshgrid(x, y):
    xx, yy = np.meshgrid(np.arange(0, x), np.arange(0, y))
    return np.concatenate([np.expand_dims(xx, 2), np.expand_dims(yy, 2)], axis=2)


def xywh2xyxy(boxes):
    xy = boxes[..., :2]
    wh = boxes[..., 2:]
    return np.concatenate([xy - wh / 2, xy + wh / 2], -1)


def xyxy2xywh(boxes):
    xymin = boxes[..., :2]
    xymax = boxes[..., 2:]
    return np.concatenate([(xymin + xymax) / 2, xymax - xymin + 1], -1)


def box_iou(box1, box2):
    x11, y11, x12, y12 = np.split(box1, 4, axis=1)
    x21, y21, x22, y22 = np.split(box2, 4, axis=1)
    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))
    interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
    iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)
    return iou


class CharDictionary:
    def __init__(self, chars):
        self._char2idx = {"<PAD>": 0, "<UNK>": 1}
        self._idx2char = ["<PAD>", "<UNK>"]
        for c in chars:
            idx = len(self._idx2char)
            self._char2idx[c] = idx
            self._idx2char.append(c)
        self._char2idx["<BLANK>"] = len(self._idx2char)
        self._idx2char.append("<BLANK>")
        self.pad_value = 0
        self.unknown_value = 1
        self.blank_value = self.char2idx("<BLANK>")

    def char2idx(self, c):
        return self._char2idx.get(c, self.unknown_value)

    def idx2char(self, i):
        if i < len(self._idx2char):
            return self._idx2char[i]
        return "<UNK>"


def _has_file_allowed_extension(filename: str, extensions: List[str]):
    filename = filename.lower()
    return any(filename.endswith(ext) for ext in extensions)


class Dataset(data.Dataset):
    def __init__(
            self,
            root,
            chardict,
            transform=None,
            image_extensions=["jpg", "jpeg", "png"],
    ):
        super().__init__()
        self.root = os.path.expanduser(root)
        self.chardict = chardict
        self.transform = transform
        self._gather_files(image_extensions)

    def _gather_files(self, image_extensions):
        candidates = [os.path.join(self.root, f) for f in os.listdir(self.root) if _has_file_allowed_extension(f, image_extensions)]
        samples = []
        for candidate in candidates:
            index = candidate.rfind(".")
            if index < 0:
                continue
            annot = candidate[:index] + ".json"
            if os.path.exists(annot):
                samples.append((candidate, annot))
        self.samples = samples

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
        length_of_longest_text = max((len(box['text']) for box in annot['boxes']))
        padded_texts = torch.zeros((len(annot['boxes']), length_of_longest_text), dtype=torch.int32)
        padded_texts[...] = self.chardict.pad_value
        for i, box in enumerate(annot['boxes']):
            text = [self.chardict.char2idx(c) for c in box['text']]
            padded_texts[i, :len(text)] = torch.tensor(text)
        boxes = torch.zeros((len(annot['boxes']), 4), dtype=torch.int32)
        for i, box in enumerate(annot['boxes']):
            boxes[i, 0] = box['left']
            boxes[i, 1] = box['top']
            boxes[i, 2] = box['left'] + box['width']
            boxes[i, 3] = box['top'] + box['height']
        return image, boxes, padded_texts

    def collate_fn(self, data):
        batch_size = len(data)
        images, boxes, texts = zip(*data)
        images = torch.stack(images, dim=0)
        max_box = max((len(bs) for bs in boxes))
        padded_boxes = torch.zeros((batch_size, max_box, 4), dtype=torch.int32)
        max_text_length = max((text.size()[-1] for text in texts))
        padded_texts = torch.zeros((batch_size, max_box, max_text_length))
        padded_texts[...] = self.chardict.pad_value
        for i, bs in enumerate(boxes):
            padded_boxes[i, :len(bs), :] = bs
        for i, ts in enumerate(texts):
            padded_texts[i, :len(ts), :ts.size()[-1]] = ts
        return images, padded_boxes, padded_texts


