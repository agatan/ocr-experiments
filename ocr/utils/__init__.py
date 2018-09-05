# %%

import numpy as np


def meshgrid(x: int, y: int) -> np.ndarray:
    xx, yy = np.meshgrid(np.arange(0, x), np.arange(0, y))
    return np.concatenate(
        [np.expand_dims(xx, axis=2), np.expand_dims(yy, axis=2)], axis=2
    )


def xywh2xyxy(boxes: np.ndarray) -> np.ndarray:
    xy = boxes[..., :2]
    wh = boxes[..., 2:]
    return np.concatenate([xy - wh / 2, xy + wh / 2], -1)


def xyxy2xywh(boxes: np.ndarray) -> np.ndarray:
    xymin = boxes[..., :2]
    xymax = boxes[..., 2:]
    return np.concatenate([(xymin + xymax) / 2, xymax - xymin + 1], -1)


def box_iou(box1: np.ndarray, box2: np.ndarray) -> np.ndarray:
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
