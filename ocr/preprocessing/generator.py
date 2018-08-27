import os
import csv
from collections import defaultdict
from typing import Tuple

import numpy as np

from ocr.utils import meshgrid
from ocr.utils.image import resize_image, read_image


class Generator(object):
    def __init__(self, input_size=(512, 512), features_pixel=8):
        self.input_size = input_size
        assert input_size[0] % features_pixel == 0
        assert input_size[1] % features_pixel == 0
        self.feature_pixel = features_pixel
        self.feature_size = (input_size[0] // features_pixel, input_size[1] // features_pixel)

    def resize_entry(self, image: np.ndarray, annots: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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

    def load_annotation(self, image_index: int) -> np.ndarray:
        raise NotImplementedError()

    def compute_ground_truth(self, annots: np.ndarray) -> np.ndarray:
        """Returns ground truth matrix.

        Args:
            annots: list of annotations ([#box, (xmin, ymin, xmax, ymax, angle)])
        Returns:
            ground_truth: [H of feature_size, W of feature_size, (confidence, left, top, right, bottom, angle)]
        """
        ground_truth = np.zeros(self.feature_size + (6,))
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
            ground_truth[ymin_ceil:ymax_floor, xmin_ceil:xmax_floor, 1] = np.arange(xmin_ceil, xmax_floor) - xmin
            ground_truth[ymin_ceil:ymax_floor, xmin_ceil:xmax_floor, 2] = np.expand_dims(np.arange(ymin_ceil, ymax_floor), axis=1) - ymin
            ground_truth[ymin_ceil:ymax_floor, xmin_ceil:xmax_floor, 3] = xmax - np.arange(xmin_ceil, xmax_floor)
            ground_truth[ymin_ceil:ymax_floor, xmin_ceil:xmax_floor, 4] = ymax - np.expand_dims(np.arange(ymin_ceil, ymax_floor), axis=1)
            ground_truth[ymin_ceil:ymax_floor, xmin_ceil:xmax_floor, 5] = annot[4]
        return ground_truth

    def batches(self, batch_size=32, infinite=True):
        while True:
            indices = np.random.permutation(self.size())
            n_batches = (self.size() - 1) // batch_size + 1
            for batch in range(n_batches):
                start_idx = batch * batch_size
                end_idx = min((batch + 1) * batch_size, self.size())
                targets = indices[start_idx:end_idx]
                images = np.zeros((len(targets),) + self.input_size + (3,))
                gts = np.zeros((len(targets),) + self.feature_size + (6,))
                for i, target in enumerate(targets):
                    image, annots = self.load_image(target), self.load_annotation(target)
                    image, annots = self.resize_entry(image, annots)
                    gt = self.compute_ground_truth(annots)
                    images[i, ...] = image / 255.0
                    gts[i, ...] = gt
                yield images, gts
            if not infinite:
                break


def _read_annotations(csvpath: str):
    with open(csvpath, 'r') as fp:
        result = defaultdict(list)
        reader = csv.DictReader(fp)
        for i, row in enumerate(reader):
            rrow = {col: float(row[col]) for col in ['xmin', 'xmax', 'ymin', 'ymax', 'angle']}
            if rrow['xmax'] <= rrow['xmin'] :
                raise ValueError(f"line {i}: xmax ({rrow['xmax']}) must be greater than xmin ({rrow['xmin']})")
            if rrow['ymax'] <= rrow['ymin'] :
                raise ValueError(f"line {i}: ymax ({rrow['ymax']}) must be greater than ymin ({rrow['ymin']})")
            result[row['image']].append(rrow)
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
        result = np.zeros((len(annots), 5)) # [#box, (xmin, ymin, xmax, ymax, angle)]
        for idx, annot in enumerate(annots):
            result[idx, 0] = annot['xmin']
            result[idx, 1] = annot['ymin']
            result[idx, 2] = annot['xmax']
            result[idx, 3] = annot['ymax']
            result[idx, 4] = annot['angle'] / 90.0
        return result
