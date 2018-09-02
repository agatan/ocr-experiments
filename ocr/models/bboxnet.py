import math

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Input, Conv2D, Lambda, Activation, BatchNormalization, LeakyReLU, \
    MaxPooling2D, Dense

from ocr.preprocessing import generator

K = tf.keras.backend


def __flatten_and_mask(y_true, y_pred):
    y_true = tf.reshape(y_true, shape=(-1, 6))
    y_pred = tf.reshape(y_pred, shape=(-1, 6))
    mask = tf.not_equal(y_true[:, 0], -1.0)
    y_true = tf.boolean_mask(y_true, mask=mask)
    y_pred = tf.boolean_mask(y_pred, mask=mask)
    return y_true, y_pred


def __loss_confidence(y_true, y_pred):
    alpha = 0.25
    gamma = 2.
    y_true = y_true[:, 0]
    y_pred = tf.nn.sigmoid(y_pred[:, 0])
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    loss_sum = -K.sum(
        alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1 + K.epsilon())
    ) - K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))
    return loss_sum / tf.cast(tf.shape(y_pred)[0], tf.float32)


def _metric_confidence_accuracy(y_true, y_pred):
    y_true, y_pred = __flatten_and_mask(y_true, y_pred)
    y_true = y_true[..., 0]
    y_pred = tf.nn.sigmoid(y_pred[..., 0])
    return tf.keras.metrics.binary_accuracy(y_true, y_pred)


def _metric_loss_confidence(y_true, y_pred):
    y_true, y_pred = __flatten_and_mask(y_true, y_pred)
    return __loss_confidence(y_true, y_pred)


def _ious(y_true, y_pred):
    mask = tf.equal(y_true[..., 0], 1.0)
    y_true = tf.boolean_mask(y_true, mask=mask)
    y_pred = tf.boolean_mask(y_pred, mask=mask)
    area_true = tf.multiply(
        y_true[..., 3] + y_true[..., 1], y_true[..., 4] + y_true[..., 2]
    )
    area_pred = tf.maximum(
        tf.multiply(y_pred[..., 3] + y_pred[..., 1], y_pred[..., 4] + y_pred[..., 2]), 0
    )
    x_intersect = tf.maximum(
        tf.minimum(y_true[..., 3], y_pred[..., 3])
        + tf.minimum(y_true[..., 1], y_pred[..., 1]),
        0,
    )
    y_intersect = tf.maximum(
        tf.minimum(y_true[..., 4], y_pred[..., 4])
        + tf.minimum(y_true[..., 2], y_pred[..., 2]),
        0,
    )
    area_intersect = tf.multiply(x_intersect, y_intersect)
    ious = area_intersect / (
        area_true + area_pred - area_intersect + tf.keras.backend.epsilon()
    )
    return ious


def __loss_iou(y_true, y_pred):
    ious = _ious(y_true, y_pred)
    return -tf.reduce_mean(tf.log(ious + tf.keras.backend.epsilon()))


def _metric_iou(y_true, y_pred):
    return tf.reduce_mean(_ious(y_true, y_pred))


def __loss_angle(y_true, y_pred):
    mask = tf.equal(y_true[..., 0], 1.0)
    y_true = tf.boolean_mask(y_true, mask=mask)
    y_pred = tf.tanh(tf.boolean_mask(y_pred, mask=mask))
    diff_angles = (y_true[..., 5] - y_pred[..., 5]) * 90 / 180 * math.pi
    return tf.reduce_mean(1 - tf.cos(diff_angles))


def _metric_loss_angle(y_true, y_pred):
    return __loss_angle(y_true, y_pred)


def _loss(y_true, y_pred):
    y_true, y_pred = __flatten_and_mask(y_true, y_pred)
    loss_confidence = __loss_confidence(y_true, y_pred)
    loss_iou = __loss_iou(y_true, y_pred)
    loss_angle = __loss_angle(y_true, y_pred)
    loss = loss_confidence + loss_iou + loss_angle
    return loss


def _reconstruct_boxes(boxes, features_pixel=8):
    batch = tf.shape(boxes)[0]
    width = boxes.shape[2]
    height = boxes.shape[1]

    xx = tf.tile(
        tf.expand_dims(
            tf.expand_dims(tf.cast(tf.range(0, width), tf.float32), axis=0), axis=0
        ),
        (batch, height, 1),
    )
    left = xx - boxes[:, :, :, 0]
    right = xx + boxes[:, :, :, 2]
    yy = tf.tile(
        tf.expand_dims(
            tf.expand_dims(tf.cast(tf.range(0, height), tf.float32), axis=0), axis=-1
        ),
        (batch, 1, width),
    )
    top = yy - boxes[:, :, :, 1]
    bottom = yy + boxes[:, :, :, 3]
    return features_pixel * tf.stack([left, top, right, bottom], axis=-1)


def _bilinear_interpolate(img: tf.Tensor, x: tf.Tensor, y: tf.Tensor):
    """
    Args:
        img: [H, W, C]
        x: [-1]
        y: [-1]
    Returns:
        interpolated_image: tf.Tensor. shape: (y, x, C)
    """

    def to_f(t):
        return tf.cast(t, dtype=tf.float32)

    x0 = tf.cast(tf.floor(x), dtype=tf.int32)
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), dtype=tf.int32)
    y1 = y0 + 1

    x0 = tf.clip_by_value(x0, 0, tf.shape(img)[1] - 1)
    x1 = tf.clip_by_value(x1, 0, tf.shape(img)[1] - 1)
    y0 = tf.clip_by_value(y0, 0, tf.shape(img)[0] - 1)
    y1 = tf.clip_by_value(y1, 0, tf.shape(img)[0] - 1)

    img_a = tf.gather(tf.gather(img, x0, axis=1), y0, axis=0)
    img_b = tf.gather(tf.gather(img, x1, axis=1), y0, axis=0)
    img_c = tf.gather(tf.gather(img, x0, axis=1), y1, axis=0)
    img_d = tf.gather(tf.gather(img, x1, axis=1), y1, axis=0)

    def meshgrid_distance(x_distance, y_distance):
        x, y = tf.meshgrid(x_distance, y_distance)
        return x * y

    wa = meshgrid_distance(to_f(x1) - x, to_f(y1) - y)
    wb = meshgrid_distance(to_f(x1) - x, y - to_f(y0))
    wc = meshgrid_distance(x - to_f(x0), to_f(y1) - y)
    wd = meshgrid_distance(x - to_f(x0), y - to_f(y0))
    wa = tf.expand_dims(wa, 2)
    wb = tf.expand_dims(wb, 2)
    wc = tf.expand_dims(wc, 2)
    wd = tf.expand_dims(wd, 2)

    flatten_interporated = img_a * wa + img_b * wb + img_c * wc + img_d * wd
    return flatten_interporated


_ROI_HEIGHT = 8


def _roi_pooling(images, boxes):
    max_width = tf.cast(
        tf.ceil(
            tf.reduce_max((boxes[:, 2] - boxes[:, 0]) / (boxes[:, 3] - boxes[:, 1]))
            * _ROI_HEIGHT
        ),
        tf.int32,
    )
    widths = tf.to_int32(tf.ceil((boxes[:, 2] - boxes[:, 0]) / (boxes[:, 3] - boxes[:, 1]) * _ROI_HEIGHT))
    widths = tf.expand_dims(widths, -1)

    def mapper(i):
        box = boxes[i]
        base_width = tf.to_float(box[2] - box[0])
        base_height = tf.to_float(box[3] - box[1])
        height = tf.to_float(_ROI_HEIGHT)
        width = tf.ceil(base_width / base_height * height)
        map_w = base_width / (width - 1)
        map_h = base_height / (height - 1)
        xx = tf.to_float(tf.range(0, tf.to_int32(width))) * map_w + box[0]
        yy = tf.to_float(tf.range(0, tf.to_int32(height))) * map_h + box[1]
        pooled = _bilinear_interpolate(images[i], xx, yy)
        padded = tf.pad(pooled, [[0, 0], [0, max_width - tf.to_int32(width)], [0, 0]])
        return padded

    indices = tf.range(tf.shape(images)[0])
    return tf.map_fn(mapper, indices, dtype=tf.float32), widths


def _ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return tf.keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)


def _text_recognition(images, n_vocab):
    x = images
    for _ in range(int(math.log2(_ROI_HEIGHT))):
        x = Conv2D(256, 3, padding='same', dilation_rate=2)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = MaxPooling2D((2, 1))(x)
    x = Dense(n_vocab, activation='softmax')(x)
    return Lambda(lambda x: tf.squeeze(x, 1))(x)


def create_model(backborn, features_pixel, input_shape=(512, 512, 3), n_vocab=10):
    image = Input(shape=input_shape, name="image")
    sampled_text_region = Input(shape=(5,), name="sampled_text_region")
    labels = Input(shape=(generator.MAX_LENGTH,), name='sampled_text', dtype=tf.float32)
    label_length = Input(shape=(1,), name='label_length', dtype=tf.int64)

    x = backborn(image)
    bbox_output = Conv2D(6, kernel_size=1, name='bbox')(x)

    # RoI Pooling and OCR
    roi, widths = Lambda(lambda args: _roi_pooling(args[0], args[1]))([x, sampled_text_region])
    smashed = _text_recognition(roi, n_vocab)

    ctc_loss = Lambda(_ctc_lambda_func, output_shape=(1,), name='ctc')([smashed, labels, widths, label_length])

    training_model = Model([image, sampled_text_region, labels, label_length], [bbox_output, ctc_loss])
    training_model.compile(
        "adam",
        loss={'bbox': _loss, 'ctc': lambda y_true, y_pred: y_pred},
        metrics={'bbox': [
            _metric_confidence_accuracy,
            _metric_iou,
            _metric_loss_confidence,
            __loss_iou,
            _metric_loss_angle,
        ]},
    )
    return training_model


def create_prediction_model(model, features_pixel):
    image = model.input
    x = model.output
    confidence = Activation("sigmoid")(
        Lambda(lambda x: x[..., 0:1], name="confidence")(x)
    )
    boxes = Lambda(
        lambda x: _reconstruct_boxes(x[..., 1:5], features_pixel=features_pixel),
        name="box",
    )(x)
    angles = Activation("tanh")(Lambda(lambda x: x[..., 5:6])(x))
    prediction_model = Model(image, [confidence, boxes, angles])
    return prediction_model


def extract_boxes(scores, boxes, angle, thres=0.9):
    scores = np.reshape(scores, (-1))
    boxes = np.reshape(boxes, (-1, 4))
    angle = np.reshape(angle, (-1))

    indices = scores > thres
    scores = scores[indices]
    boxes = boxes[indices]
    angle = angle[indices]
    return scores, boxes, angle
