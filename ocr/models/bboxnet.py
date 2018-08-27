import math

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Input, Conv2D, Lambda, Activation

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
    loss_sum = -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1 + K.epsilon()))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))
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
    area_true = tf.multiply(y_true[..., 3] + y_true[..., 1], y_true[..., 4] + y_true[..., 2])
    area_pred = tf.maximum(tf.multiply(y_pred[..., 3] + y_pred[..., 1], y_pred[..., 4] + y_pred[..., 2]), 0)
    x_intersect = tf.maximum(tf.minimum(y_true[..., 3], y_pred[..., 3]) + tf.minimum(y_true[..., 1], y_pred[..., 1]), 0)
    y_intersect = tf.maximum(tf.minimum(y_true[..., 4], y_pred[..., 4]) + tf.minimum(y_true[..., 2], y_pred[..., 2]), 0)
    area_intersect = tf.multiply(x_intersect, y_intersect)
    ious = area_intersect / (area_true + area_pred - area_intersect + tf.keras.backend.epsilon())
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
    loss = loss_confidence + loss_iou  + loss_angle
    return loss


def _reconstruct_boxes(boxes,  features_pixel=8):
    batch = tf.shape(boxes)[0]
    width = boxes.shape[2]
    height = boxes.shape[1]

    xx = tf.tile(tf.expand_dims(tf.expand_dims(tf.cast(tf.range(0, width), tf.float32), axis=0), axis=0),
                 (batch, height, 1))
    left = xx - boxes[:, :, :, 0]
    right = xx + boxes[:, :, :, 2]
    yy = tf.tile(tf.expand_dims(tf.expand_dims(tf.cast(tf.range(0, height), tf.float32), axis=0), axis=-1),
                 (batch, 1, width))
    top = yy - boxes[:, :, :, 1]
    bottom = yy + boxes[:, :, :, 3]
    return features_pixel *  tf.stack([left, top, right, bottom], axis=-1)


def create_model(backborn, features_pixel, input_shape=(512, 512, 3)):
    image = Input(shape=input_shape, name='image')
    x = backborn(image)
    output = Conv2D(6, kernel_size=1)(x)
    training_model = Model(image, output)
    training_model.compile('adam', loss=_loss, metrics=[_metric_confidence_accuracy, _metric_iou, _metric_loss_confidence, __loss_iou, _metric_loss_angle])
    return training_model


def create_prediction_model(model, features_pixel):
    image = model.input
    x = model.output
    confidence = Activation('sigmoid')(Lambda(lambda x: x[..., 0:1], name='confidence')(x))
    boxes = Lambda(lambda x: _reconstruct_boxes(x[..., 1:5], features_pixel=features_pixel), name='box')(x)
    angles = Activation('tanh')(Lambda(lambda x: x[..., 5:6])(x))
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
