import os
import csv
import random
from collections import defaultdict
from typing import Tuple, List

import numpy as np
from imgaug import augmenters as iaa

from ocr.data import process
from ocr.utils.image import resize_image, read_image


MAX_LENGTH = 64


class Generator(object):
    def __init__(self, input_size=(512, 512), features_pixel=8, aug=False):
        self.input_size = input_size
        assert input_size[0] % features_pixel == 0
        assert input_size[1] % features_pixel == 0
        self.feature_pixel = features_pixel
        self.feature_size = (
            input_size[0] // features_pixel,
            input_size[1] // features_pixel,
        )
        if aug:
            self.aug = iaa.Sequential(
                [
                    iaa.ContrastNormalization((0.75, 1.5)),
                    iaa.AdditiveGaussianNoise(
                        loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
                    ),
                    iaa.Multiply((0.8, 1.2), per_channel=0.2),
                    iaa.Add((-10, 10), per_channel=0.5),
                    iaa.Pepper((0, 0.05), per_channel=0.2),
                    iaa.GaussianBlur((0, 2.0)),
                ]
            )
        else:
            self.aug = None

    def resize_entry(
        self, image: np.ndarray, annots: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        resized_image = resize_image(image, self.input_size)
        annots = annots.copy()
        x_ratio = resized_image.shape[1] / image.shape[1]
        y_ratio = resized_image.shape[0] / image.shape[0]
        annots[:, 0] *= x_ratio
        annots[:, 2] *= x_ratio
        annots[:, 1] *= y_ratio
        annots[:, 3] *= y_ratio
        return resized_image, annots

    def size(self) -> int:
        raise NotImplementedError()

    def load_image(self, image_index: int) -> np.ndarray:
        raise NotImplementedError()

    def load_annotation(self, image_index: int) -> Tuple[np.ndarray, List[str]]:
        raise NotImplementedError()

    def char2idx(self, c) -> int:
        raise NotImplementedError()

    def compute_ground_truth(self, annots: np.ndarray) -> np.ndarray:
        """Returns ground truth matrix.

        Args:
            annots: list of annotations ([#box, (xmin, ymin, xmax, ymax)])
        Returns:
            ground_truth: [H of feature_size, W of feature_size, (confidence, left, top, right, bottom)]
        """
        ground_truth = np.zeros(self.feature_size + (5,))
        for annot in annots:
            xmin = annot[0] / self.feature_pixel
            xmin_ceil = np.maximum(np.math.ceil(xmin), 0)
            xmin_floor = np.maximum(np.math.floor(xmin), 0)
            xmax = annot[2] / self.feature_pixel
            xmax_ceil = np.minimum(np.math.ceil(xmax), self.feature_size[1])
            xmax_floor = np.minimum(np.math.floor(xmax), self.feature_size[1])
            ymin = annot[1] / self.feature_pixel
            ymin_ceil = np.maximum(np.math.ceil(ymin), 0)
            ymin_floor = np.maximum(np.math.floor(ymin), 0)
            ymax = annot[3] / self.feature_pixel
            ymax_ceil = np.minimum(np.math.ceil(ymax), self.feature_size[0])
            ymax_floor = np.minimum(np.math.floor(ymax), self.feature_size[0])
            ground_truth[ymin_floor:ymin_ceil, xmin_floor:xmin_ceil, 0] = -1
            ground_truth[ymax_floor:ymax_ceil, xmax_floor:xmax_ceil, 0] = -1
            ground_truth[ymin_ceil:ymax_floor, xmin_ceil:xmax_floor, 0] = 1
            ground_truth[ymin_ceil:ymax_floor, xmin_ceil:xmax_floor, 1] = (
                np.arange(xmin_ceil, xmax_floor) - xmin
            )
            ground_truth[ymin_ceil:ymax_floor, xmin_ceil:xmax_floor, 2] = (
                np.expand_dims(np.arange(ymin_ceil, ymax_floor), axis=1) - ymin
            )
            ground_truth[
                ymin_ceil:ymax_floor, xmin_ceil:xmax_floor, 3
            ] = xmax - np.arange(xmin_ceil, xmax_floor)
            ground_truth[
                ymin_ceil:ymax_floor, xmin_ceil:xmax_floor, 4
            ] = ymax - np.expand_dims(np.arange(ymin_ceil, ymax_floor), axis=1)
        return ground_truth

    def batches(self, batch_size=32, infinite=True):
        while True:
            indices = np.random.permutation(self.size())
            n_batches = (self.size() - 1) // batch_size + 1
            for batch in range(n_batches):
                start_idx = batch * batch_size
                end_idx = min((batch + 1) * batch_size, self.size())
                targets = indices[start_idx:end_idx]
                images = np.zeros(
                    (len(targets),) + self.input_size + (3,), dtype=np.uint8
                )
                gts = np.zeros((len(targets),) + self.feature_size + (5,))
                sample_text_regions = np.zeros((len(targets), 4))
                sample_text = np.zeros((len(targets), MAX_LENGTH), dtype=np.int32)
                label_length = np.zeros((len(targets)), dtype=np.int64)
                for i, target in enumerate(targets):
                    image, (annots, texts) = (
                        self.load_image(target),
                        self.load_annotation(target),
                    )
                    image, annots = self.resize_entry(image, annots)
                    sample_annot_index = random.randint(0, len(annots) - 1)
                    sample_annot = (
                        annots[sample_annot_index].astype("float32")
                        / self.feature_pixel
                    )
                    sample_text_annot = np.array(
                        [self.char2idx(c) for c in texts[sample_annot_index]]
                    )
                    sample_text_regions[i, ...] = sample_annot
                    sample_text[i, : len(sample_text_annot)] = sample_text_annot
                    label_length[i] = len(sample_text_annot)
                    gt = self.compute_ground_truth(annots)
                    images[i, ...] = image
                    gts[i, ...] = gt
                if self.aug:
                    images = self.aug.augment_images(images)
                images = images.astype(np.float32) / 255.0
                yield [images, sample_text_regions, sample_text, label_length], {
                    "bbox": gts,
                    "ctc": np.zeros(len(targets)),
                }
            if not infinite:
                break


def _read_annotations(csvpath: str):
    with open(csvpath, "r") as fp:
        result = defaultdict(list)
        reader = csv.DictReader(fp)
        for i, row in enumerate(reader):
            rrow = {col: float(row[col]) for col in ["xmin", "xmax", "ymin", "ymax"]}
            if rrow["xmax"] <= rrow["xmin"]:
                raise ValueError(
                    f"line {i}: xmax ({rrow['xmax']}) must be greater than xmin ({rrow['xmin']})"
                )
            if rrow["ymax"] <= rrow["ymin"]:
                raise ValueError(
                    f"line {i}: ymax ({rrow['ymax']}) must be greater than ymin ({rrow['ymin']})"
                )
            rrow["text"] = row["text"]
            result[row["image"]].append(rrow)
    return result


class CSVGenerator(Generator):
    def __init__(self, csv_file_path: str, basedir: str = None, **kwargs):
        self.basedir = basedir
        if self.basedir is None:
            self.basedir = os.path.dirname(csv_file_path)
        self.annotations = _read_annotations(csv_file_path)
        self.image_names = list(self.annotations.keys())
        super().__init__(**kwargs)

    def _image_path(self, image_index: int) -> str:
        return os.path.join(self.basedir, self.image_names[image_index])

    def size(self) -> int:
        return len(self.image_names)

    def load_image(self, image_index: int) -> np.ndarray:
        return read_image(self._image_path(image_index))

    def load_annotation(self, image_index: int) -> np.ndarray:
        annots = self.annotations[self.image_names[image_index]]
        result = np.zeros((len(annots), 4))  # [#box, (xmin, ymin, xmax, ymax)]
        text = []
        for idx, annot in enumerate(annots):
            result[idx, 0] = annot["xmin"]
            result[idx, 1] = annot["ymin"]
            result[idx, 2] = annot["xmax"]
            result[idx, 3] = annot["ymax"]
            text.append(annot["text"].replace("\n", ""))
        return result, text

    def char2idx(self, c):
        return process.char2idx(c)
