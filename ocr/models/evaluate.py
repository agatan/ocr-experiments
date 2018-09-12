import os
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import matplotlib.pyplot as plt
from ocr.preprocessing.generator import CSVGenerator
from ocr.models import resnet50, bboxnet_subclass, mobilenet
from ocr.data import process

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
estimator = tf.estimator.Estimator(bboxnet_subclass.model_fn, model_dir='./checkpoints')

import random
import cv2
import math
import numpy as np
from itertools import groupby

x, y = next(CSVGenerator('./data/processed/annotations.csv', features_pixel=8, input_size=(512 // 2, 832 // 2)).batches(4))
input_fn = tf.estimator.inputs.numpy_input_fn(x, shuffle=False)

images = x['image']

hooks = [tf_debug.LocalCLIDebugHook()]

for image, predict in zip(images, estimator.predict(input_fn, hooks=hooks)):
    boxes = predict['boxes']
    texts = predict['texts']
    print(boxes)
    print(texts)
    target = (image.copy() * 255).astype(np.uint8)
    for (l, t, r, b) in boxes:
        cv2.rectangle(target, (l, t), (r, b), (255, 0, 0), thickness=3)
    plt.imshow(target)
    plt.show()
