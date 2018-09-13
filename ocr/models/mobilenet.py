from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.applications import MobileNet
from tensorflow.python.keras.layers import (
    Conv2D,
    BatchNormalization,
    UpSampling2D,
    Add,
    LeakyReLU,
    DepthwiseConv2D,
    MaxPooling2D, Lambda, Dense, ReLU, ZeroPadding2D)
import tensorflow as tf


K = tf.keras.backend


def _conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1), training=False):
    """Convolution block in mobilenet from keras-applications. (modified for estimator API)
    """
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    filters = int(filters * alpha)
    x = ZeroPadding2D(padding=(1, 1), name='conv1_pad')(inputs)
    x = Conv2D(
        filters,
        kernel,
        padding='valid',
        use_bias=False,
        strides=strides,
        name='conv1')(x)
    x = BatchNormalization(axis=channel_axis, name='conv1_bn')(x, training=training)
    return ReLU(6, name='conv1_relu')(x)


def _depthwise_conv_block(inputs, pointwise_conv_filters, alpha, depth_multiplier=2, strides=(1, 1), block_id=1, training=False):
    """Depthwise Separable Convolution block in mobilenet from keras-applications. (modified for estimator API)
    """
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)
    x = ZeroPadding2D(padding=(1, 1), name='conv_pad_%d' % block_id)(inputs)
    x = DepthwiseConv2D(  # pylint: disable=not-callable
        (3, 3),
        padding='valid',
        depth_multiplier=depth_multiplier,
        strides=strides,
        use_bias=False,
        name='conv_dw_%d' % block_id)(x)
    x = BatchNormalization(axis=channel_axis, name='conv_dw_%d_bn' % block_id)(x, training=training)
    x = ReLU(6, name='conv_dw_%d_relu' % block_id)(x)

    x = Conv2D(
        pointwise_conv_filters, (1, 1),
        padding='same',
        use_bias=False,
        strides=(1, 1),
        name='conv_pw_%d' % block_id)(
        x)
    x = BatchNormalization(axis=channel_axis, name='conv_pw_%d_bn' % block_id)(x, training=training)
    return ReLU(6, name='conv_pw_%d_relu' % block_id)(x)


def _deconv_block(x, filters, kernel_size=1, training=False):
    x = DepthwiseConv2D(kernel_size=kernel_size, padding="same", use_bias=False)(x)
    x = BatchNormalization()(x, training=training)
    x = LeakyReLU()(x)
    x = Conv2D(filters, kernel_size=1, use_bias=False)(x)
    x = BatchNormalization()(x, training=training)
    x = LeakyReLU()(x)
    return UpSampling2D()(x)


def backbone(input_tensor, training=True, alpha=1.0, depth_multiplier=1):
    x = _conv_block(input_tensor, 32, alpha, strides=(2, 2))
    x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=1, training=training)
    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, strides=(2, 2), block_id=2, training=training)
    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=3, training=training)
    x = _depthwise_conv_block(x, 256, alpha, depth_multiplier, strides=(2, 2), block_id=4, training=training)
    x = _depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=5, training=training)
    l256 = x
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, strides=(2, 2), block_id=6, training=training)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=7, training=training)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=8, training=training)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=9, training=training)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=10, training=training)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=11, training=training)
    l512 = x
    x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier, strides=(2, 2,), block_id=12, training=training)
    x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier, block_id=13, training=training)
    x = _deconv_block(x, 512, kernel_size=3, training=training)
    x = x + l512
    x = _deconv_block(x, 256, kernel_size=3, training=training)
    x = x + l256
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