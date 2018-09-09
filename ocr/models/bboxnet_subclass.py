import tensorflow as tf
from tensorflow.python.estimator.model_fn import ModeKeys
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Conv2D, Lambda, SeparableConv2D, ReLU, MaxPooling2D, Dropout, Dense

from ocr.data import process
from ocr.models import mobilenet
from ocr.preprocessing import generator

K = tf.keras.backend


def model_fn(features, labels, mode, params):
    training = mode == ModeKeys.TRAIN
    tf.keras.backend.set_learning_phase(training)
    image = features['image']
    bbox_true = labels['bbox']
    # sampled_text_region, text, text_length = labels['sampled_text_region'], labels['text'], labels['text_length']
    text_regions, texts, text_lengths = labels['text_regions'], labels['texts'], labels['text_lengths']
    # setup models
    backbone, features_pixel = mobilenet.backbone()
    text_recognition_horizontal = _text_recognition_horizontal_model(
        input_shape=(None, None, backbone.output.shape[-1]), n_vocab=process.vocab())
    text_recognition_vertical = _text_recognition_vertical_model(input_shape=(None, None, backbone.output.shape[-1]),
                                                                 n_vocab=process.vocab())

    # run
    fmap = backbone(image, training=training)
    bbox_output = Conv2D(5, kernel_size=1, name='bbox')(fmap)

    if mode == ModeKeys.TRAIN or mode == ModeKeys.PREDICT:
        # loss
        ctc_loss = _crop_and_ocr(fmap, text_regions, features_pixel, text_recognition_horizontal, text_recognition_vertical, mode=mode, labels=texts, label_lengths=text_lengths)
        iou = _metric_iou(bbox_true, bbox_output)
        accuracy = _metric_confidence_accuracy(bbox_true, bbox_output)
        bbox_loss = _loss_bbox(bbox_true, bbox_output)
        loss = ctc_loss + bbox_loss
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = tf.train.AdamOptimizer().minimize(loss, global_step=tf.train.get_global_step())

        tf.summary.scalar('iou', iou)
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('loss/bbox', bbox_loss)
        tf.summary.scalar('loss/ctc', ctc_loss)

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    scores = tf.nn.sigmoid(bbox_output[..., 0])
    bounding_boxes = _reconstruct_boxes(bbox_output[..., 1:5], features_pixel=features_pixel)
    nms_boxes = _nms(bounding_boxes, scores)
    text = _crop_and_ocr(fmap, nms_boxes, features_pixel, text_recognition_horizontal, text_recognition_vertical, mode=mode)
    return tf.estimator.EstimatorSpec(mode=mode, predictions={
        'boxes': nms_boxes,
        'texts': text,
    })


def make_input_fn(gen, batch_size=8):
    def input_fn():
        return tf.data.Dataset.from_generator(lambda: gen.batches(batch_size, infinite=True),
                                              output_types=({'image': tf.float32},
                                                            {'bbox': tf.float32, 'text_regions': tf.float32,
                                                             'texts': tf.int32, 'text_lengths': tf.int32}),
                                              output_shapes=({'image': tf.TensorShape([None, None, None, 3])},
                                                             {'bbox': tf.TensorShape([None, None, None, 5]), 'text_regions': tf.TensorShape([None, generator.MAX_BOX, 4]),
                                                              'texts': tf.TensorShape([None, generator.MAX_BOX, None]), 'text_lengths': tf.TensorShape([None, generator.MAX_BOX])})).prefetch(32)

    return input_fn


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


def _roi_pooling_horizontal(images, boxes):
    # width >= height?
    is_horizontal = tf.greater_equal(
        boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    )
    boxes = tf.where(is_horizontal, boxes, tf.zeros_like(boxes))
    non_zero_boxes = tf.logical_or(
        tf.greater_equal(boxes[:, 2] - boxes[:, 0], 0.1),
        tf.greater_equal(boxes[:, 3] - boxes[:, 1], 0.1),
    )
    nanable_ratios = (boxes[:, 2] - boxes[:, 0]) / (boxes[:, 3] - boxes[:, 1])
    ratios = tf.where(non_zero_boxes, nanable_ratios, tf.zeros_like(nanable_ratios))
    widths = tf.to_int32(tf.ceil(ratios * _ROI_HEIGHT))
    max_width = tf.reduce_max(widths)
    widths = tf.expand_dims(widths, -1)

    def mapper(i):
        box = boxes[i]
        base_width = tf.to_float(box[2] - box[0])
        base_height = tf.to_float(box[3] - box[1])

        def cond():
            return tf.logical_or(
                tf.greater_equal(base_width, 0.1), tf.greater_equal(base_height, 0.1)
            )

        def non_zero():
            height = tf.to_float(_ROI_HEIGHT)
            width = tf.ceil(base_width / base_height * height)
            map_w = base_width / (width - 1)
            map_h = base_height / (height - 1)
            xx = tf.to_float(tf.range(0, tf.to_int32(width))) * map_w + box[0]
            yy = tf.to_float(tf.range(0, tf.to_int32(height))) * map_h + box[1]
            pooled = _bilinear_interpolate(images[i], xx, yy)
            padded = tf.pad(
                pooled, [[0, 0], [0, max_width - tf.to_int32(width)], [0, 0]]
            )
            return padded

        def zero():
            return tf.zeros((_ROI_HEIGHT, max_width, tf.shape(images)[-1]))

        return tf.cond(cond(), non_zero, zero)

    indices = tf.range(tf.shape(images)[0])
    return tf.map_fn(mapper, indices, dtype=tf.float32), widths


_ROI_WIDTH = 8


def _roi_pooling_vertical(images, boxes):
    # width < height?
    is_vertical = tf.less(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])
    boxes = tf.where(is_vertical, boxes, tf.zeros_like(boxes))
    non_zero_boxes = tf.logical_or(
        tf.greater_equal(boxes[:, 2] - boxes[:, 0], 0.1),
        tf.greater_equal(boxes[:, 3] - boxes[:, 1], 0.1),
    )
    nanable_ratios = (boxes[:, 3] - boxes[:, 1]) / (boxes[:, 2] - boxes[:, 0])
    ratios = tf.where(non_zero_boxes, nanable_ratios, tf.zeros_like(nanable_ratios))
    heights = tf.cast(tf.ceil(ratios * _ROI_WIDTH), tf.int32)
    max_height = tf.reduce_max(heights)
    heights = tf.expand_dims(heights, -1)

    def mapper(i):
        box = boxes[i]
        base_width = tf.to_float(box[2] - box[0])
        base_height = tf.to_float(box[3] - box[1])

        def cond():
            return tf.logical_or(
                tf.greater_equal(base_width, 0.1), tf.greater_equal(base_height, 0.1)
            )

        def non_zero():
            width = tf.to_float(_ROI_WIDTH)
            height = tf.ceil(base_height / base_width * width)
            map_w = base_width / (width - 1)
            map_h = base_height / (height - 1)
            xx = tf.to_float(tf.range(0, tf.to_int32(width))) * map_w + box[0]
            yy = tf.to_float(tf.range(0, tf.to_int32(height))) * map_h + box[1]
            pooled = _bilinear_interpolate(images[i], xx, yy)
            padded = tf.pad(
                pooled, [[0, max_height - tf.to_int32(height)], [0, 0], [0, 0]]
            )
            return padded

        def zero():
            return tf.zeros((max_height, _ROI_WIDTH, tf.shape(images)[-1]))

        return tf.cond(cond(), non_zero, zero)

    indices = tf.range(tf.shape(images)[0])
    return tf.map_fn(mapper, indices, dtype=tf.float32), heights


def _text_recognition_horizontal_model(input_shape, n_vocab):
    roi = Input(shape=input_shape, name="roi_horizontal")
    x = roi
    for c in [64, 128, 256]:
        x = SeparableConv2D(c, 3, padding="same")(x)
        # TODO(agatan): if input_shape contains 0, GroupNormalization can generate nan weights.
        # x = GroupNormalization()(x)
        x = ReLU(6.)(x)
        x = SeparableConv2D(c, 3, padding="same")(x)
        # x = GroupNormalization()(x)
        x = ReLU(6.)(x)
        x = MaxPooling2D((2, 1))(x)
    x = Lambda(lambda v: tf.squeeze(v, 1))(x)
    x = Dropout(0.2)(x)
    output = Dense(n_vocab, activation="softmax")(x)
    return Model(roi, output, name="horizontal_model")


def _text_recognition_vertical_model(input_shape, n_vocab):
    roi = Input(shape=input_shape, name="roi_vertical")
    x = roi
    for c in [64, 128, 256]:
        x = SeparableConv2D(c, 3, padding="same")(x)
        # TODO(agatan): if input_shape contains 0, GroupNormalization can generate nan weights.
        # x = GroupNormalization()(x)
        x = ReLU(6.)(x)
        x = SeparableConv2D(c, 3, padding="same")(x)
        # x = GroupNormalization()(x)
        x = ReLU(6.)(x)
        x = MaxPooling2D((1, 2))(x)
    x = Lambda(lambda v: tf.squeeze(v, 2))(x)
    x = Dropout(0.2)(x)
    output = Dense(n_vocab, activation="softmax")(x)
    return Model(roi, output, name="vertical_model")


def _pad_horizontal_and_vertical(args):
    horizontal, vertical = args
    maximum = tf.maximum(tf.shape(horizontal)[1], tf.shape(vertical)[1])
    horizontal = tf.pad(
        horizontal, [[0, 0], [0, maximum - tf.shape(horizontal)[1]], [0, 0]]
    )
    vertical = tf.pad(vertical, [[0, 0], [0, maximum - tf.shape(vertical)[1]], [0, 0]])
    return horizontal, vertical


## Loss functions

def _ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return tf.keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)


def __flatten_and_mask(y_true, y_pred):
    y_true = tf.reshape(y_true, shape=(-1, 5))
    y_pred = tf.reshape(y_pred, shape=(-1, 5))
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


def _loss_bbox(y_true, y_pred):
    y_true, y_pred = __flatten_and_mask(y_true, y_pred)
    loss_confidence = __loss_confidence(y_true, y_pred)
    loss_iou = __loss_iou(y_true, y_pred)
    loss = loss_confidence + loss_iou
    return loss


def _crop_and_ocr(images, boxes, features_pixel, text_recognition_horizontal_model, text_recognition_vertical_model, mode, labels=None, label_lengths=None):
    training = mode != ModeKeys.PREDICT
    if training:
        assert labels is not None and label_lengths is not None
    boxes = boxes / features_pixel
    ratios = (boxes[..., 2] - boxes[..., 0]) / (boxes[..., 3] - boxes[..., 1])
    vertical_ratios = 1 / ratios
    non_zero_boxes = tf.logical_or(
        tf.greater_equal(boxes[..., 2] - boxes[..., 0], 0.1),
        tf.greater_equal(boxes[..., 3] - boxes[..., 1], 0.1),
    )
    ratios = tf.where(non_zero_boxes, ratios, tf.zeros_like(ratios))
    vertical_ratios = tf.where(
        non_zero_boxes, vertical_ratios, tf.zeros_like(vertical_ratios)
    )
    max_width = tf.to_int32(tf.ceil(tf.reduce_max(ratios * _ROI_HEIGHT)))
    max_height = tf.to_int32(tf.ceil(tf.reduce_max(vertical_ratios * _ROI_WIDTH)))
    max_length = tf.maximum(max_width, max_height)

    def _mapper(i):
        bbs = boxes[:, i, :]
        roi_horizontal, widths = _roi_pooling_horizontal(images, bbs)
        smashed_horizontal = text_recognition_horizontal_model(roi_horizontal, training=training)
        roi_vertical, heights = _roi_pooling_vertical(images, bbs)
        smashed_vertical = text_recognition_vertical_model(roi_vertical, training=training)
        widths = tf.squeeze(widths, axis=-1)
        heights = tf.squeeze(heights, axis=-1)
        smashed_horizontal, smashed_vertical = _pad_horizontal_and_vertical(
            [smashed_horizontal, smashed_vertical]
        )
        smashed = tf.where(
            tf.not_equal(widths, 0), smashed_horizontal, smashed_vertical
        )
        lengths = tf.where(tf.not_equal(widths, 0), widths, heights)
        cond = tf.not_equal(tf.shape(smashed)[1], 0)

        def then_branch():
            if not training:
                decoded, _probas = tf.keras.backend.ctc_decode(
                    smashed, lengths, greedy=False
                )
                return tf.pad(
                    decoded[0],
                    [[0, 0], [0, max_length - tf.shape(decoded[0])[1]]],
                    constant_values=-1,
                )
            else:
                indices = tf.squeeze(tf.where(tf.not_equal(label_lengths[:, i], 0)), axis=-1)
                ls = tf.gather(labels[:, i, :], indices)
                ss = tf.gather(smashed, indices)
                input_lengths = tf.gather(lengths, indices)
                l_lengths = tf.gather(label_lengths[:, i], indices)
                return tf.reduce_sum(tf.keras.backend.ctc_batch_cost(ls, ss, tf.expand_dims(input_lengths, axis=-1), tf.expand_dims(l_lengths, axis=-1)))

        def else_branch():
            if not training:
                return -tf.ones((tf.shape(bbs)[0], max_length), dtype=tf.int64)
            else:
                return tf.zeros(())

        return tf.cond(cond, then_branch, else_branch)

    if not training:
        text_recognition = tf.map_fn(_mapper, tf.range(0, generator.MAX_BOX), dtype=tf.int64)
        return tf.transpose(text_recognition, [1, 0, 2])
    else:
        return tf.reduce_sum(tf.map_fn(_mapper, tf.range(0, generator.MAX_BOX), dtype=tf.float32))


def _nms(boxes, scores):
    def mapper(i):
        bbs, ss = boxes[i], scores[i]
        bbs = tf.reshape(bbs, [-1, 4])
        ss = tf.reshape(ss, [-1])
        indices = tf.image.non_max_suppression(bbs, ss, generator.MAX_BOX)
        bbs = tf.gather(bbs, indices)
        ss = tf.gather(ss, indices)
        return tf.where(tf.greater_equal(ss, 0.5), bbs, tf.zeros_like(bbs))
    idx = tf.range(0, tf.shape(boxes)[0])
    return tf.map_fn(mapper, idx, dtype=tf.float32)


def _reconstruct_boxes(boxes, features_pixel=8):
    boxes_shape = tf.shape(boxes)
    batch = boxes_shape[0]
    width = boxes_shape[2]
    height = boxes_shape[1]

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