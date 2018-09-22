import tensorflow as tf

from ocr.data import process
from ocr.models import mobilenet
from ocr.preprocessing import generator
from ocr.preprocessing.generator import CSVGenerator


class _TextRecognition(tf.keras.Model):
    def __init__(self, pool_size, n_vocab, data_format):
        super(_TextRecognition, self).__init__()
        assert pool_size in [(2, 1), (1, 2)]
        self.is_horizontal = pool_size == (2, 1)

        def _conv_block(channels):
            return tf.keras.Sequential([
                tf.keras.layers.Conv2D(channels, 3, padding='same', use_bias=False, data_format=data_format),
                tf.keras.layers.ReLU(max_value=6.),
                tf.keras.layers.Conv2D(channels, 3, padding='same', use_bias=False, data_format=data_format),
                tf.keras.layers.ReLU(max_value=6.),
                tf.keras.layers.MaxPooling2D(pool_size=pool_size, data_format=data_format)
            ])
        self.conv1 = _conv_block(64)
        self.conv2 = _conv_block(128)
        self.conv3 = _conv_block(256)
        self.dense = tf.keras.Sequential([
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(n_vocab, activation='softmax')
        ])

    def call(self, inputs, training=True):
        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        x = tf.squeeze(x, 1 if self.is_horizontal else 2)
        x = self.dense(x, training=training)
        return x


class OCRTrainNet(tf.keras.Model):
    def __init__(self, backbone_cls, n_vocab, data_format=None):
        super(OCRTrainNet, self).__init__()
        self.backbone = backbone_cls(data_format=data_format)
        self.feature_map_scale = backbone_cls.feature_map_scale()

        self.bbox_head = tf.keras.layers.Conv2D(5, kernel_size=3, padding='same', use_bias=True, name='bbox')
        self.horizontal_text_recognition_branch = _TextRecognition((2, 1), n_vocab=n_vocab, data_format=data_format)
        self.vertical_text_recognition_branch = _TextRecognition((1, 2), n_vocab=n_vocab, data_format=data_format)

    def call(self, inputs, training=True):
        image, bboxes, text_boxes, text_sequences, text_lengths = inputs
        fmap = self.backbone(image, training=training)
        bbox_pred = self.bbox_head(fmap)
        confidence_loss, bbox_loss = _loss_bbox(y_true=bboxes, y_pred=bbox_pred)
        ocr_loss = self._ocr_loss(fmap, text_boxes, text_sequences, text_lengths)
        return confidence_loss, bbox_loss, ocr_loss

    def _ocr_loss(self, fmap, text_boxes, text_sequences, text_lengths):
        text_boxes /= self.feature_map_scale

        def mapper(i):
            box = text_boxes[:, i, :]
            horizontal_roi, widths = _roi_pooling_horizontal(fmap, box)
            horizontal_text_recog = self.horizontal_text_recognition_branch(horizontal_roi, training=True)
            vertical_roi, heights = _roi_pooling_vertical(fmap, box)
            vertical_text_recog = self.vertical_text_recognition_branch(vertical_roi, training=True)
            widths = tf.squeeze(widths, axis=-1)
            heights = tf.squeeze(heights, axis=-1)
            horizontal_text_recog, vertical_text_recog = _pad_horizontal_and_vertical(horizontal_text_recog, vertical_text_recog)
            text_recog = tf.where(
                tf.not_equal(widths, 0), horizontal_text_recog, vertical_text_recog,
            )
            lengths = tf.where(tf.not_equal(widths, 0), widths, heights)
            cond = tf.not_equal(tf.shape(text_recog)[1], 0)

            def true_fn():
                indices = tf.squeeze(tf.where(tf.not_equal(text_lengths[:, i], 0)), axis=-1)
                labels = tf.gather(text_sequences[:, i, :], indices)
                recogs = tf.gather(text_recog, indices)
                input_lengths = tf.gather(lengths, indices)
                label_lengths = tf.gather(text_lengths[:, i], indices)
                return tf.reduce_mean(
                    tf.keras.backend.ctc_batch_cost(
                        labels, recogs,
                        tf.expand_dims(input_lengths, axis=-1),
                        tf.expand_dims(label_lengths, axis=-1),
                    ),
                )

            def false_fn():
                return tf.zeros(())

            return tf.cond(cond, true_fn, false_fn)

        return tf.reduce_sum(tf.map_fn(mapper, tf.range(0, generator.MAX_BOX), dtype=tf.float32))


class OCRNet(tf.keras.Model):
    def __init__(self, ocr_train_net):
        super(OCRNet, self).__init__()
        self.backbone = ocr_train_net.backbone
        self.feature_map_scale = ocr_train_net.feature_map_scale
        self.bbox_conv = ocr_train_net.bbox_head
        self.horizontal_text_recoginition_branch = ocr_train_net.horizontal_text_recognition_branch
        self.vertical_text_recoginition_branch = ocr_train_net.vertical_text_recognition_branch

    def call(self, inputs, training=False):
        fmap = self.backbone(inputs, training=training)
        bbox_output = self.bbox_conv(fmap)
        scores = tf.nn.sigmoid(bbox_output[..., 0])
        bounding_boxes = _reconstruct_boxes(bbox_output[..., 1:5], feature_map_scale=self.feature_map_scale)
        nms_boxes = _nms(bounding_boxes, scores)
        text = self._crop_and_ocr(fmap, nms_boxes, training=training)
        return nms_boxes, text

    def _crop_and_ocr(self, fmap, boxes, training=False):
        boxes = boxes / self.feature_map_scale
        horizontal_ratios = (boxes[..., 2] - boxes[..., 0]) / (boxes[..., 3] - boxes[..., 1])
        vertical_ratios = (boxes[..., 3] - boxes[..., 1]) / (boxes[..., 2] - boxes[..., 0])
        non_zero_boxes = tf.logical_or(
            tf.greater_equal(boxes[..., 2] - boxes[..., 0], 0.001),
            tf.greater_equal(boxes[..., 3] - boxes[..., 1], 0.001),
        )
        horizontal_ratios = tf.where(non_zero_boxes, horizontal_ratios, tf.zeros_like(horizontal_ratios))
        vertical_ratios = tf.where(non_zero_boxes, vertical_ratios, tf.zeros_like(vertical_ratios))
        max_width = tf.to_int32(tf.ceil(tf.reduce_max(horizontal_ratios * _ROI_HEIGHT)))
        max_height = tf.to_int32(tf.ceil(tf.reduce_max(vertical_ratios * _ROI_HEIGHT)))
        max_length = tf.maximum(max_width, max_height)

        def mapper(i):
            box = boxes[:, i, :]
            horizontal_roi, widths = _roi_pooling_horizontal(fmap, box)
            horizontal_recog = self.horizontal_text_recoginition_branch(horizontal_roi, training=training)
            vertical_roi, heights = _roi_pooling_vertical(fmap, box)
            vertical_recog = self.vertical_text_recoginition_branch(vertical_roi, training=training)
            widths = tf.squeeze(widths, axis=-1)
            heights = tf.squeeze(heights, axis=-1)
            horizontal_recog, vertical_recog = _pad_horizontal_and_vertical(horizontal_recog, vertical_recog)
            recog = tf.where(
                tf.not_equal(widths, 0), horizontal_recog, vertical_recog,
            )
            lengths = tf.where(tf.not_equal(widths, 0), widths, heights)
            cond = tf.not_equal(tf.shape(recog)[1], 0)

            def true_fn():
                decoded, _ = tf.keras.backend.ctc_decode(recog, lengths, greedy=False)
                return tf.pad(
                    decoded[0],
                    [[0, 0], [0, max_length - tf.shape(decoded[0])[1]]],
                    constant_values=-1,
                )

            def false_fn():
                return -tf.ones((tf.shape(box)[0], max_length), dtype=tf.int64)

            return tf.cond(cond, true_fn, false_fn)

        text_recognition = tf.map_fn(mapper, tf.range(0, generator.MAX_BOX), dtype=tf.int64)
        return tf.transpose(text_recognition, [1, 0, 2])


def __loss_confidence(y_true, y_pred):
    K = tf.keras.backend
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
    if tf.executing_eagerly():
        print(tf.reduce_mean(ious).numpy())
    return ious


def __loss_iou(y_true, y_pred):
    ious = _ious(y_true, y_pred)
    return -tf.reduce_mean(tf.log(ious + tf.keras.backend.epsilon()))


def __flatten_and_mask(y_true, y_pred):
    y_true = tf.reshape(y_true, shape=(-1, 5))
    y_pred = tf.reshape(y_pred, shape=(-1, 5))
    mask = tf.not_equal(y_true[:, 0], -1.0)
    y_true = tf.boolean_mask(y_true, mask=mask)
    y_pred = tf.boolean_mask(y_pred, mask=mask)
    return y_true, y_pred


def _loss_bbox(y_true, y_pred):
    y_true, y_pred = __flatten_and_mask(y_true, y_pred)
    loss_confidence = __loss_confidence(y_true, y_pred)
    loss_iou = __loss_iou(y_true, y_pred)
    return loss_confidence, loss_iou


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
    base_widths = tf.where(tf.less_equal(boxes[:, 2] - boxes[:, 0], 0.0), 1e-4 * tf.ones_like(boxes[:, 2]), boxes[:, 2] - boxes[:, 0])
    base_heights = tf.where(tf.less_equal(boxes[:, 3] - boxes[:, 1], 0.0), 1e-4 * tf.ones_like(boxes[:, 3]), boxes[:, 3] - boxes[:, 1])
    ratios = tf.where(non_zero_boxes, base_widths / base_heights, tf.zeros_like(base_widths))
    # nanable_ratios = (boxes[:, 2] - boxes[:, 0]) / (boxes[:, 3] - boxes[:, 1])
    # ratios = tf.where(non_zero_boxes, nanable_ratios, tf.zeros_like(nanable_ratios))
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
    roi = tf.map_fn(mapper, indices, dtype=tf.float32)
    roi.set_shape([None, 8, None, images.shape[-1]])
    return roi, widths


def _roi_pooling_vertical(images, boxes):
    # width < height?
    is_vertical = tf.less(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])
    boxes = tf.where(is_vertical, boxes, tf.zeros_like(boxes))
    non_zero_boxes = tf.logical_or(
        tf.greater_equal(boxes[:, 2] - boxes[:, 0], 0.1),
        tf.greater_equal(boxes[:, 3] - boxes[:, 1], 0.1),
    )
    # nanable_ratios = (boxes[:, 3] - boxes[:, 1]) / (boxes[:, 2] - boxes[:, 0])
    # ratios = tf.where(non_zero_boxes, nanable_ratios, tf.zeros_like(nanable_ratios))
    base_widths = tf.where(tf.less_equal(boxes[:, 2] - boxes[:, 0], 0.0), 1e-4 * tf.ones_like(boxes[:, 2]), boxes[:, 2] - boxes[:, 0])
    base_heights = tf.where(tf.less_equal(boxes[:, 3] - boxes[:, 1], 0.0), 1e-4 * tf.ones_like(boxes[:, 3]), boxes[:, 3] - boxes[:, 1])
    ratios = tf.where(non_zero_boxes, base_heights / base_widths, tf.zeros_like(base_widths))
    heights = tf.cast(tf.ceil(ratios * _ROI_HEIGHT), tf.int32)
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
            width = tf.to_float(_ROI_HEIGHT)
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
            return tf.zeros((max_height, _ROI_HEIGHT, tf.shape(images)[-1]))

        return tf.cond(cond(), non_zero, zero)

    indices = tf.range(tf.shape(images)[0])
    roi = tf.map_fn(mapper, indices, dtype=tf.float32)
    roi.set_shape([None, None, 8, images.shape[-1]])
    return roi, heights


def _pad_horizontal_and_vertical(horizontal, vertical):
    maximum = tf.maximum(tf.shape(horizontal)[1], tf.shape(vertical)[1])
    horizontal = tf.pad(
        horizontal, [[0, 0], [0, maximum - tf.shape(horizontal)[1]], [0, 0]]
    )
    vertical = tf.pad(vertical, [[0, 0], [0, maximum - tf.shape(vertical)[1]], [0, 0]])
    return horizontal, vertical


def _nms(boxes, scores):
    def mapper(i):
        bbs, ss = boxes[i], scores[i]
        bbs = tf.reshape(bbs, [-1, 4])
        ss = tf.reshape(ss, [-1])
        indices = tf.image.non_max_suppression(bbs, ss, generator.MAX_BOX)
        bbs = tf.gather(bbs, indices)
        ss = tf.gather(ss, indices)
        return tf.where(tf.greater_equal(ss, 0.3), bbs, tf.zeros_like(bbs))
    idx = tf.range(0, tf.shape(boxes)[0])
    return tf.map_fn(mapper, idx, dtype=tf.float32)


def _reconstruct_boxes(boxes, feature_map_scale=8):
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
    return feature_map_scale * tf.stack([left, top, right, bottom], axis=-1)


def model_fn(features, labels, mode):
    image = features['image']
    train_model = OCRTrainNet(mobilenet.MobileNetV2Backbone, process.vocab())
    model = OCRNet(train_model)

    training = mode == tf.estimator.ModeKeys.TRAIN
    if mode != tf.estimator.ModeKeys.PREDICT:
        conf_loss, bbox_loss, ocr_loss = train_model(
            [image, labels['bbox'], labels['text_regions'], labels['texts'], labels['text_lengths']], training=training)
        loss = conf_loss + bbox_loss + ocr_loss
        with tf.control_dependencies(train_model.get_updates_for(image)):
            train_op = tf.train.AdamOptimizer().minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
        )
    boxes, texts = model(image)
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions={
            'boxes': boxes,
            'texts': texts,
        }
    )


def main_eager():
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    tf.enable_eager_execution(config=config)
    gen = CSVGenerator(
        './data/processed/annotations.csv', features_pixel=mobilenet.MobileNetV2Backbone.feature_map_scale(), input_size=(192 // 2, 256 // 2)
    )
    dataset = tf.data.Dataset.from_generator(lambda: gen.batches(1, infinite=True),
                                   output_types=({'image': tf.float32},
                                                 {'bbox': tf.float32,
                                                  'text_regions': tf.float32,
                                                  'texts': tf.int32,
                                                  'text_lengths': tf.int32}),
                                   output_shapes=({'image': tf.TensorShape([None, None, None, 3])},
                                                   {'bbox': tf.TensorShape([None, None, None, 5]),
                                                   'text_regions': tf.TensorShape([None, generator.MAX_BOX, 4]),
                                                   'texts': tf.TensorShape([None, generator.MAX_BOX, None]),
                                                   'text_lengths': tf.TensorShape([None, generator.MAX_BOX])}))
    train_model = OCRTrainNet(mobilenet.MobileNetV2Backbone, n_vocab=process.vocab())
    model = OCRNet(train_model)
    optimizer = tf.train.AdamOptimizer()
    global_step = tf.train.get_or_create_global_step()
    checkpointer = tf.train.Checkpoint(
        train_model=train_model,
        model=model,
        optimizer=optimizer,
        global_step=global_step,
    )
    latest_ckpt = tf.train.latest_checkpoint('/tmp/ocr/')
    checkpointer.restore(latest_ckpt)
    print("Restore from checkpoint: {}".format(latest_ckpt))

    if False:
        for x, y in dataset:
            nms_boxes, texts = model(x['image'], training=False)
            img = (x['image'][0] * 255).numpy().astype(np.uint8)
            nms_boxes = nms_boxes[0].numpy().astype(np.int32)
            print(nms_boxes.shape)
            import cv2
            for l, t, r, b in nms_boxes:
                cv2.rectangle(img, (l, t), (r, b), (255, 0, 0))
            import matplotlib.pyplot as plt
            for text in texts[0].numpy():
                print(''.join(process.idx2char(c) if c != -1 else '' for c in text))
            plt.imshow(img)
            plt.show()
            break
    for step, (x, y) in enumerate(dataset):
        with tf.GradientTape() as tape:
            conf_loss, bbox_loss, ocr_loss = train_model([x['image'], y['bbox'], y['text_regions'], y['texts'], y['text_lengths']], training=True)
            loss = conf_loss + bbox_loss + ocr_loss
            grad = tape.gradient(loss, train_model.variables)
            optimizer.apply_gradients(zip(grad, train_model.variables), global_step=global_step)
        print('Step: {}, Loss: {:.3f} (Confidence: {:.3f}, Box: {:.3f}, OCR: {:.3f})'.format(global_step.numpy(), loss, conf_loss, bbox_loss, ocr_loss))
        if step % 10 == 0:
            checkpointer.save('/tmp/ocr/ckpt')
        if step == 1000:
            break


def main():
    gen = CSVGenerator(
        './data/processed/annotations.csv', features_pixel=mobilenet.MobileNetV2Backbone.feature_map_scale(), input_size=(192 // 4, 256 // 4)
    )
    def input_fn():
        return tf.data.Dataset.from_generator(lambda: gen.batches(1, infinite=True),
                                   output_types=({'image': tf.float32},
                                                 {'bbox': tf.float32,
                                                  'text_regions': tf.float32,
                                                  'texts': tf.int32,
                                                  'text_lengths': tf.int32}),
                                   output_shapes=({'image': tf.TensorShape([None, None, None, 3])},
                                                   {'bbox': tf.TensorShape([None, None, None, 5]),
                                                   'text_regions': tf.TensorShape([None, generator.MAX_BOX, 4]),
                                                   'texts': tf.TensorShape([None, generator.MAX_BOX, None]),
                                                   'text_lengths': tf.TensorShape([None, generator.MAX_BOX])}))
    estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir='/tmp/ocr')
    estimator.train(input_fn, max_steps=100)
    estimator.evaluate(input_fn, steps=16)


if __name__ == '__main__':
    main_eager()
