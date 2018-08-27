from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.applications import MobileNet
from tensorflow.python.keras.layers import (
    Conv2D,
    BatchNormalization,
    ReLU,
    UpSampling2D,
    Add,
)


def _deconv_block(x, filters):
    x = Conv2D(filters, kernel_size=1)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return UpSampling2D()(x)


def backbone(input_shape=(512, 512, 3)):
    image = Input(shape=input_shape, name="image")
    base_model = MobileNet(input_tensor=image, include_top=False, weights=None)
    x = base_model.output
    x = Conv2D(512, kernel_size=3, padding="same", activation="relu")(x)
    x = Conv2D(256, kernel_size=3, padding="same", activation="relu")(x)
    model = Model(image, x)
    return model, 32
