from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.applications import MobileNet
from tensorflow.python.keras.layers import (
    Conv2D,
    BatchNormalization,
    UpSampling2D,
    Add,
    LeakyReLU,
    DepthwiseConv2D,
    MaxPooling2D, Lambda, Dense)
import tensorflow as tf


def _deconv_block(x, filters, kernel_size=1):
    x = DepthwiseConv2D(kernel_size=kernel_size, padding="same", use_bias=False)(x)
    x = tf.layers.BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters, kernel_size=1, use_bias=False)(x)
    x = tf.layers.BatchNormalization()(x)
    x = LeakyReLU()(x)
    return UpSampling2D()(x)


def _build_mobilenet_without_keras_bn(input_tensor, training):
    base_model = MobileNet(input_tensor=input_tensor, include_top=False, weights=None)
    layers = [l for l in base_model.layers]
    x = base_model.input
    for l in layers:
        if type(l) is BatchNormalization:
            x = tf.layers.BatchNormalization()(x, training=training)
        else:
            x = l(x)
    return x


def backbone(input_tensor, training=True):
    x = _build_mobilenet_without_keras_bn(input_tensor, training=training)
    # base_model = _build_mobilenet_without_keras_bn(input_shape)
    # x = base_model(input_tensor)
    x = _deconv_block(x, 512, kernel_size=3)
    # y = base_model.layers[81].output
    # x = Add()([x, y])
    x = _deconv_block(x, 256, kernel_size=3)
    # y = base_model.layers[39].output
    # x = Add()([x, y])
    return x


def features_pixel():
    return 8


def backbone1(input_tensor, training=True):
    x = MaxPooling2D()(input_tensor)
    x = MaxPooling2D()(x)
    x = tf.layers.BatchNormalization()(x, training=training)
    x = MaxPooling2D()(x)
    x = Dense(256)(x)
    return x, []