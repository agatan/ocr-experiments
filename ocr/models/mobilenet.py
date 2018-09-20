import tensorflow as tf


def _conv_bn(out, strides, data_format):
    axis = 1 if data_format == "channels_first" else 3
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(out, 3, strides=strides, padding='same', use_bias=False, data_format=data_format),
        tf.keras.layers.BatchNormalization(axis=axis),
        tf.keras.layers.ReLU(max_value=6),
    ])


def _conv1x1_bn(out, data_format):
    axis = 1 if data_format == "channels_first" else 3
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(out, 1, use_bias=False, data_format=data_format),
        tf.keras.layers.BatchNormalization(axis=axis),
        tf.keras.layers.ReLU(max_value=6),
    ])


class _InvertedResidual(tf.keras.Model):
    def __init__(self, inp, out, strides, expand_ratio, data_format=None):
        super(_InvertedResidual, self).__init__()
        assert strides in [1, 2]
        self._strides = strides
        hidden_dim = round(inp * expand_ratio)
        self._out = out
        self._use_res_connect = strides == 1 and inp == out
        self._data_format = data_format or 'channels_last'
        assert data_format in ['channels_first', 'channels_last']
        axis = 1 if data_format == "channels_first" else 3

        if expand_ratio == 1:
            self.conv = tf.keras.Sequential([
                # depth-wise
                tf.keras.layers.DepthwiseConv2D(3, strides=strides, padding='same', use_bias=False, data_format=data_format),
                tf.keras.layers.BatchNormalization(axis=axis),
                tf.keras.layers.ReLU(max_value=6),
                # point-wise
                tf.keras.layers.Conv2D(out, 1, strides=1, use_bias=False, data_format=data_format),
                tf.keras.layers.BatchNormalization(axis=axis),
            ])
        else:
            self.conv = tf.keras.Sequential([
                # point-wise
                tf.keras.layers.Conv2D(hidden_dim, 1, strides=1, use_bias=False, data_format=data_format),
                tf.keras.layers.BatchNormalization(axis=axis),
                tf.keras.layers.ReLU(max_value=6),
                # depth-wise
                tf.keras.layers.DepthwiseConv2D(3, strides=strides, padding='same', use_bias=False, data_format=data_format),
                tf.keras.layers.BatchNormalization(axis=axis),
                tf.keras.layers.ReLU(max_value=6),
                # point-wise
                tf.keras.layers.Conv2D(out, 1, strides=1, use_bias=False, data_format=data_format),
                tf.keras.layers.BatchNormalization(axis=axis),
            ])

    def call(self, inputs, training=True):
        if self._use_res_connect:
            return inputs + self.conv(inputs, training=training)
        return self.conv(inputs, training=training)

    def compute_output_shape(self, input_shape):
        if self._data_format == 'channels_last':
            batch_size, height, width, channels = input_shape
        else:
            batch_size, channels, height, width = input_shape
        if self._strides == 2:
            height = height // tf.Dimension(2)
            width = width // tf.Dimension(2)
        if self._data_format == 'channels_last':
            return tf.TensorShape([batch_size, height, width, self._out])
        else:
            return tf.TensorShape([batch_size, self._out, height, width])


def _upconv(out, data_format):
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(out, kernel_size=1, data_format=data_format, use_bias=False),
        tf.keras.layers.UpSampling2D(),
    ])


class MobileNetV2Backbone(tf.keras.Model):
    def __init__(self, data_format=None):
        super(MobileNetV2Backbone, self).__init__()
        data_format = data_format or 'channels_last'
        assert data_format in ['channels_first', 'channels_last']
        input_channel = 32
        inverted_residual_config = [
            # t (expand ratio), channel, n (layers), stride
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        self.conv1 = _conv_bn(input_channel, 2, data_format=data_format)
        for i, (t, c, n, s) in enumerate(inverted_residual_config):
            output_channel = c
            layers = []
            for j in range(n):
                if j == 0:
                    layers.append(_InvertedResidual(input_channel, output_channel, s, expand_ratio=t, data_format=data_format))
                else:
                    layers.append(_InvertedResidual(input_channel, output_channel, 1, expand_ratio=t, data_format=data_format))
                input_channel = output_channel
            setattr(self, f'block{i}', tf.keras.Sequential(layers))
        self.last_channel = 1280
        self.conv2 = _conv1x1_bn(self.last_channel, data_format=data_format)
        self.upconv3 = _upconv(96, data_format=data_format)
        self.upconv2 = _upconv(32, data_format=data_format)
        self.upconv1 = _upconv(24, data_format=data_format)

    def call(self, inputs, training=True):
        x = self.conv1(inputs, training=training)
        x = self.block0(x, training=training)
        p1 = x = self.block1(x, training=training)
        p2 = x = self.block2(x, training=training)
        x = self.block3(x, training=training)
        p3 = x = self.block4(x, training=training)
        x = self.block5(x, training=training)
        x = self.block6(x, training=training)
        x = self.conv2(x, training=training)
        x = p3 + self.upconv3(x, training=training)
        x = p2 + self.upconv2(x, training=training)
        x = p1 + self.upconv1(x, training=training)
        return x

    @staticmethod
    def feature_map_scale():
        return 4
