from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import (
    Input,
    UpSampling2D,
    Conv2D,
    BatchNormalization,
    Add,
    ReLU,
)
from tensorflow.python.keras.applications import ResNet50


def _deconv_block(x, filters):
    x = Conv2D(filters, kernel_size=1)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return UpSampling2D()(x)


def backbone(input_shape=(512, 512, 3)):
    image = Input(shape=input_shape, name="image")
    base_model = ResNet50(input_tensor=image, include_top=False, weights=None)
    x = base_model.get_layer(name="activation_48").output
    x = _deconv_block(x, 1024)
    x = Add()([x, base_model.get_layer(name="activation_39").output])
    x = _deconv_block(x, 512)
    x = Add()([x, base_model.get_layer(name="activation_21").output])
    model = Model(image, x)
    return model, 8
