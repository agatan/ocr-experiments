import cv2
import numpy as np


def overlay_annotations(image: np.ndarray, annots: np.ndarray) -> np.ndarray:
    image = image.copy()
    for annot in annots.astype(int):
        cv2.rectangle(image, (annot[0], annot[1]), (annot[2], annot[3]), (255, 0, 0), 3)
    return image
