from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.applications import MobileNet
from tensorflow.python.keras.layers import (
    Conv2D,
    BatchNormalization,
    UpSampling2D,
    Add,
    LeakyReLU,
    DepthwiseConv2D,
)


def _deconv_block(x, filters, kernel_size=1):
    x = DepthwiseConv2D(kernel_size=kernel_size, padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters, kernel_size=1, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    return UpSampling2D()(x)


def backbone(input_shape=(512, 512, 3)):
    base_model = MobileNet(input_shape=input_shape, include_top=False, weights=None)
    x = base_model.output
    x = _deconv_block(x, 512, kernel_size=3)
    y = base_model.layers[81].output
    x = Add()([x, y])
    x = _deconv_block(x, 256, kernel_size=3)
    y = base_model.layers[39].output
    x = Add()([x, y])
    model = Model(base_model.input, x)
    return model, 8
