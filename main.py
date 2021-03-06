import os
import json
import math
import tensorflow as tf
import numpy as np
from PIL import Image
import datagen
from group_norm import GroupNormalization

from icecream import ic


INPUT_SIZE = np.array([304, 192])
FEATURE_SIZE = INPUT_SIZE // 4


def meshgrid(x, y):
    xx, yy = np.meshgrid(np.arange(0, x), np.arange(0, y))
    return np.concatenate([np.expand_dims(xx, 2), np.expand_dims(yy, 2)], axis=2)


def xywh2xyxy(boxes):
    xy = boxes[..., :2]
    wh = boxes[..., 2:]
    return np.concatenate([xy - wh / 2, xy + wh / 2], -1)


def xyxy2xywh(boxes):
    xymin = boxes[..., :2]
    xymax = boxes[..., 2:]
    return np.concatenate([(xymin + xymax) / 2, xymax - xymin + 1], -1)


def box_iou(box1, box2):
    x11, y11, x12, y12 = np.split(box1, 4, axis=1)
    x21, y21, x22, y22 = np.split(box2, 4, axis=1)
    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))
    interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
    iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)
    return iou


class Sequence(tf.keras.utils.Sequence):
    def __init__(self, root: str, input_size: np.ndarray = INPUT_SIZE, batch_size=32):
        self.root = root
        self.batch_size = batch_size
        self.input_size = input_size
        self.anchor_areas = [16.0 * 16.0, 32 * 32.0, 64 * 64.0]
        self.aspect_ratios = [2.0, 8.0, 16.0]
        self.anchor_wh = self._get_anchor_wh()
        self.anchor_boxes = self._get_anchor_boxes(input_size)
        self.max_boxes = 20
        self.max_characters = 80
        self._load_data()

    def _get_anchor_wh(self):
        anchor_wh = []
        for s in self.anchor_areas:
            for ar in self.aspect_ratios:
                h = math.sqrt(s / ar)
                w = ar * h
                anchor_wh.append([w, h])
        return np.array(anchor_wh).reshape(9, 2)

    def _get_anchor_boxes(self, input_size: np.ndarray):
        fm_size = np.ceil(input_size / pow(2, 2))
        grid_size = input_size / fm_size
        fm_w, fm_h = int(fm_size[0]), int(fm_size[1])
        xy = meshgrid(fm_w, fm_h) + 0.5
        xy = np.tile((xy * grid_size).reshape(fm_h, fm_w, 1, 2), [1, 1, 9, 1])
        wh = np.tile(self.anchor_wh.reshape(1, 1, 9, 2), [fm_h, fm_w, 1, 1])
        box = np.concatenate((xy, wh), axis=3)
        return box.reshape(-1, 4)

    def _load_data(self):
        fnames = []
        bounding_boxes = []
        texts = []
        lengths = []
        data_number = 0
        while True:
            f = os.path.join(self.root, f"{data_number}.json")
            if not os.path.isfile(f):
                break
            data_number += 1
            with open(f, "r") as fp:
                info = json.load(fp)
            fnames.append(info["file"])
            boxes = np.zeros((self.max_boxes, 4))
            ts = np.zeros((self.max_boxes, self.max_characters), dtype=np.int32)
            ls = np.zeros(self.max_boxes, dtype=np.int32)
            for i, b in enumerate(info["boxes"]):
                xmin, ymin = float(b["left"]), float(b["top"])
                xmax, ymax = xmin + float(b["width"]), ymin + float(b["height"])
                boxes[i] = [xmin, ymin, xmax, ymax]
                text = datagen.text2idx(b["text"])
                assert len(text) <= self.max_characters
                ts[i, : len(text)] = text
                ls[i] = len(text)
            bounding_boxes.append(boxes)
            texts.append(ts)
            lengths.append(ls)
        self.fnames = fnames
        self.bounding_boxes = np.array(bounding_boxes)
        self.texts = np.array(texts)
        self.lengths = lengths
        self.n_examples = data_number

    def __len__(self):
        return math.ceil(self.n_examples / self.batch_size)

    def __getitem__(self, idx):
        start = self.batch_size * idx
        end = min(self.batch_size * (idx + 1), self.n_examples)
        images = np.zeros((end - start, self.input_size[1], self.input_size[0], 3))
        for i, f in enumerate(self.fnames[start:end]):
            img = Image.open(os.path.join(self.root, f))
            if img.mode != "RGB":
                img = img.convert("RGB")
            img = img.resize((304, 192))
            img = np.array(img)
            img_std = (img - img.mean()) / img.std()
            images[i] = np.array(img_std)
        box_targets = []
        for boxes in self.bounding_boxes[start:end]:
            box_targets.append(self.encode(boxes, self.input_size))
        box_targets = np.array(box_targets)

        return (
            images,
            self.bounding_boxes[start:end],
            self.texts[start:end],
            box_targets,
        )

    def generator(self):
        idx = 0
        for i, f in enumerate(self.fnames):
            img = Image.open(os.path.join(self.root, f))
            if img.mode != "RGB":
                img = img.convert("RGB")
            # img = img.resize((304, 192))
            img = np.array(img)
            img_std = (img - img.mean()) / img.std()
            position_targets = self.encode(self.bounding_boxes[i], self.input_size)
            yield img_std, self.bounding_boxes[i], self.texts[i], position_targets

    def make_input_fn(self, batch_size=16, shuffle=True):
        def input_fn():
            dataset = tf.data.Dataset.from_generator(
                self.generator,
                (tf.float32, tf.float32, tf.int32, tf.float32),
                (
                    tf.TensorShape([INPUT_SIZE[1], INPUT_SIZE[0], 3]),
                    tf.TensorShape([self.max_boxes, 4]),
                    tf.TensorShape([self.max_boxes, self.max_characters]),
                    tf.TensorShape([FEATURE_SIZE[1], FEATURE_SIZE[0], 9 * 5]),
                ),
            )
            if shuffle:
                dataset = dataset.shuffle(512)
            images, boxes, texts, positions = (
                dataset.batch(batch_size).make_one_shot_iterator().get_next()
            )
            return (
                dict(images=images),
                dict(boxes=boxes, texts=texts, positions=positions),
            )

        return input_fn

    def encode(self, boxes: np.ndarray, input_size: np.ndarray):
        """
        Args:
            boxes: np.ndarray [#box, [xmin, ymin, xmax, ymax]]
            input_size: W, H
        Returns:
        loc_targets: np.ndarray [FH, FW, #anchor * [confidence, xcenter, ycenter, width, height]]
        """
        new_boxes = []
        for points in boxes:
            if np.all(points == 0):
                break
            new_boxes.append(points)
        boxes = np.array(new_boxes)
        fm_size = [math.ceil(i / pow(2, 2)) for i in input_size]
        anchor_boxes = self._get_anchor_boxes(input_size)
        ious = box_iou(xywh2xyxy(anchor_boxes), boxes)
        boxes = xyxy2xywh(boxes)

        max_ids = np.argmax(ious, axis=1)
        max_ious = np.max(ious, axis=1)
        boxes = boxes[max_ids]

        loc_xy = (boxes[:, :2] - anchor_boxes[:, :2]) / anchor_boxes[:, 2:]
        loc_wh = np.log(boxes[:, 2:] / anchor_boxes[:, 2:])
        loc_targets = np.concatenate([loc_xy, loc_wh], axis=1)

        masks = np.ones_like(max_ids)
        masks[max_ious < 0.5] = 0

        loc_targets = loc_targets.reshape(fm_size[1], fm_size[0], 9, 4)
        masks = masks.reshape(fm_size[1], fm_size[0], 9, 1)
        return np.concatenate([masks, loc_targets], axis=3).reshape(
            fm_size[1], fm_size[0], 9 * 5
        )

    def decode(self, loc_preds, input_size: np.ndarray, conf_thres=0.5):
        anchor_boxes = self._get_anchor_boxes(input_size)
        loc_preds = loc_preds.reshape(-1, 5)
        conf_preds = loc_preds[:, 0]
        # TODO: use tensorflow. sigmoid is required.
        loc_xy = loc_preds[:, 1:3]
        loc_wh = loc_preds[:, 3:]

        xy = loc_xy * anchor_boxes[:, 2:] + anchor_boxes[:, :2]
        wh = np.exp(loc_wh) * anchor_boxes[:, 2:]
        boxes = np.concatenate([xy - wh / 2, xy + wh / 2], axis=1)

        score = conf_preds  # TODO: sigmoid
        ids = score > conf_thres
        ids = np.nonzero(ids)[0]
        return boxes[ids]


def _bottleneck(inputs, channels, strides):
    x = inputs
    x = tf.keras.layers.Conv2D(channels, kernel_size=1, use_bias=False)(x)
    x = GroupNormalization(channels // 4)(x)
    x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)
    x = tf.keras.layers.Conv2D(
        channels, kernel_size=3, strides=strides, use_bias=False, padding="same"
    )(x)
    x = GroupNormalization(channels // 4)(x)
    x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)
    x = tf.keras.layers.Conv2D(2 * channels, kernel_size=1, use_bias=False)(x)
    x = GroupNormalization(channels // 4)(x)
    if strides != 1 or inputs.shape[-1] != x.shape[-1]:
        y = tf.keras.layers.Conv2D(
            2 * channels, kernel_size=1, strides=strides, use_bias=False
        )(inputs)
        res = GroupNormalization(channels // 4)(y)
        x = tf.keras.layers.Add()([x, res])
    return tf.keras.layers.Activation(tf.keras.activations.relu)(x)


def feature_extract(inputs):
    x = tf.keras.layers.Conv2D(
        64, kernel_size=3, strides=1, padding="same", use_bias=False
    )(inputs)
    x = GroupNormalization(32)(x)
    x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)
    c1 = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding="same")(x)
    c2 = _bottleneck(c1, 64, strides=1)
    c3 = _bottleneck(c2, 128, strides=2)
    c4 = _bottleneck(c3, 256, strides=2)
    c5 = _bottleneck(c4, 512, strides=2)
    p5 = tf.keras.layers.Conv2D(256, kernel_size=1, strides=1)(c5)
    x = tf.keras.layers.Conv2D(256, kernel_size=1, strides=1)(c4)
    p4 = tf.keras.layers.Add()([tf.keras.layers.UpSampling2D(size=(2, 2))(p5), x])
    p4 = tf.keras.layers.Conv2D(128, kernel_size=3, strides=1, padding="same")(p4)
    x = tf.keras.layers.Conv2D(128, kernel_size=1, strides=1)(c3)
    p3 = tf.keras.layers.Add()([tf.keras.layers.UpSampling2D(size=(2, 2))(p4), x])
    p3 = tf.keras.layers.Conv2D(128, kernel_size=3, strides=1, padding="same")(p3)
    return p3


def position_predict(feature_map, name):
    return tf.keras.layers.Conv2D(9 * 5, kernel_size=1, strides=1, name=name)(
        feature_map
    )


def reconstruct_bounding_boxes(anchor_boxes, outputs):
    """
    Args:
        anchor_boxes: tensor
        outputs: tensor [#batch, H, W, [preds, x, y, w, h]] (x, y in 0..1)
    Returns:
        boxes: tensor [#batch, H * W, [preds, x1, y1, x2, y2]]
    """
    anchor_boxes = tf.convert_to_tensor(anchor_boxes, dtype=tf.float32)
    flatten = tf.reshape(outputs, (-1, FEATURE_SIZE[0] * FEATURE_SIZE[1] * 9, 5))
    out_xys = flatten[..., 1:3]
    out_whs = flatten[..., 3:5]

    preds = tf.sigmoid(flatten[..., 0:1])
    xy = out_xys * anchor_boxes[:, 2:] + anchor_boxes[:, :2]
    wh = tf.exp(out_whs) * anchor_boxes[:, 2:]
    xy_min = xy - wh / 2
    xy_max = xy + wh / 2
    xmin = tf.clip_by_value(xy_min[..., :1], 0, INPUT_SIZE[0])
    ymin = tf.clip_by_value(xy_min[..., 1:], 0, INPUT_SIZE[1])
    xmax = tf.clip_by_value(xy_max[..., :1], xmin, INPUT_SIZE[0])
    ymax = tf.clip_by_value(xy_max[..., 1:], ymin, INPUT_SIZE[1])
    boxes = tf.concat([preds, xmin, ymin, xmax, ymax], axis=2)

    def nms_fn(boxes):
        MAX_BOXES = 64
        indices = tf.where(boxes[:, 0] > 0.5)
        boxes = tf.gather_nd(boxes, indices)
        indices = tf.image.non_max_suppression(boxes[:, 1:], boxes[:, 0], MAX_BOXES)
        boxes = tf.gather(boxes, indices)
        boxes = tf.pad(boxes, [[0, MAX_BOXES - tf.shape(boxes)[0]], [0, 0]])
        boxes.set_shape([MAX_BOXES, 5])
        return boxes

    boxes = tf.map_fn(nms_fn, boxes, dtype=tf.float32)
    return boxes


def roi_pooling(image: tf.Tensor, boxes: tf.Tensor, height):
    base_widths = boxes[:, 2] - boxes[:, 0]
    base_heights = boxes[:, 3] - boxes[:, 1]
    aspects = base_widths / base_heights
    # widths = tf.ceil(aspects * float(height))
    max_width = 80

    def mapper(box):
        cond = tf.logical_and(
            tf.reduce_any(tf.not_equal(box, 0)),
            tf.logical_and(box[2] > box[0], box[3] > box[1]),
        )

        def then_branch():
            base_width = box[2] - box[0]
            base_height = box[3] - box[1]
            aspect = base_width / base_height
            width = tf.ceil(aspect * float(height))
            width = tf.minimum(float(max_width), width)
            map_w = base_width / (width - 1)
            map_h = base_height / (height - 1)
            xx = tf.range(0, width, dtype=tf.float32) * map_w + box[0]
            yy = tf.range(0, height, dtype=tf.float32) * map_h + box[1]
            pooled = bilinear_interpolate(image, xx, yy)
            padded = tf.pad(
                pooled, [[0, 0], [0, max_width - tf.cast(width, tf.int32)], [0, 0]]
            )
            padded.set_shape((height, max_width, None))
            return padded

        def else_branch():
            return tf.zeros((height, max_width, tf.shape(image)[-1]))

        return tf.cond(cond, then_branch, else_branch)

    results = tf.map_fn(mapper, boxes)
    results.set_shape([None, height, None, image.shape[-1]])
    return results


def roi_pooling_in_batch(feature_maps: tf.Tensor, boxes: tf.Tensor, height: int):
    def mapper(i):
        fm = feature_maps[i]
        bbs = boxes[i]
        return roi_pooling(fm, bbs, height)

    return tf.map_fn(mapper, tf.range(tf.shape(feature_maps)[0]), dtype=tf.float32)


def roi_pooling_lengths_in_batch(boxes, height):
    base_widths = boxes[..., 2] - boxes[..., 0]
    base_heights = boxes[..., 3] - boxes[..., 1]
    aspects = base_widths / base_heights
    widths = tf.ceil(aspects * float(height))
    zeroed = tf.where(tf.equal(base_heights, 0.0), tf.zeros_like(widths), widths)
    return tf.cast(tf.minimum(zeroed, 80.0), tf.int32)


def bilinear_interpolate(img: tf.Tensor, x: tf.Tensor, y: tf.Tensor):
    """
    Args:
        img: [H, W, C]
        x: [-1]
        y: [-1]
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


def ocr_predict(features):
    def mapper(feature):
        def block(x, channels):
            return tf.keras.Sequential(
                [
                    tf.keras.layers.Conv2D(
                        channels, kernel_size=3, strides=1, padding="same"
                    ),
                    GroupNormalization(groups=channels // 4),
                    tf.keras.layers.Activation(tf.keras.activations.relu),
                    tf.keras.layers.MaxPooling2D((2, 1), strides=(2, 1)),
                ]
            )(x)

        x = feature
        for c in [128, 256]:
            x = block(x, c)
        x = tf.keras.layers.Conv2D(
            len(datagen.CHAR2IDX) + 2, kernel_size=3, strides=1, padding="same"
        )(x)
        x = tf.keras.layers.Softmax()(x)
        return tf.keras.backend.squeeze(x, axis=1)

    return tf.keras.backend.map_fn(mapper, features)


def make_model_fn(anchor_boxes: np.ndarray):
    def model_fn(features, labels, mode):
        training = mode == tf.estimator.ModeKeys.TRAIN
        images = features["images"]
        feature_map = feature_extract(images)
        positions = position_predict(feature_map, name="positions")

        if mode == tf.estimator.ModeKeys.PREDICT:
            reconstructed_boxes = reconstruct_bounding_boxes(anchor_boxes, positions)
            predicted_boxes_pooled = roi_pooling_in_batch(
                feature_map, reconstructed_boxes, 4
            )
            box_lengths = roi_pooling_lengths_in_batch(reconstructed_boxes, 4)
            ocr_prediction = ocr_predict(predicted_boxes_pooled)
            decoded_ocr_predictions = decode_ocr(box_lengths, ocr_prediction)
            predictions = {
                "images": images,
                "box_confidences": reconstructed_boxes[..., 0],
                "boxes": reconstructed_boxes[..., 1:],
                "ocr": decoded_ocr_predictions,
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        true_boxes = labels["boxes"]
        true_texts = labels["texts"]
        true_positions = labels["positions"]
        pooled = roi_pooling_in_batch(feature_map, true_boxes, 4)
        box_lengths = roi_pooling_lengths_in_batch(true_boxes, 4)
        ocr_prediction = ocr_predict(pooled)

        position_loss = loss_positions(true_positions, positions)
        ocr_loss = calc_ocr_loss(box_lengths, true_texts, ocr_prediction)
        tf.summary.scalar("loss/position_loss", position_loss)
        tf.summary.scalar("loss/ocr_loss", ocr_loss)
        loss = position_loss + ocr_loss

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=dict())

        optimizer = tf.train.AdadeltaOptimizer()
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    return model_fn


def weighted_binary_cross_entropy(output, target, weights):
    loss = weights[1] * (target * tf.log(output + 1e-8)) + weights[0] * (
        (1 - target) * tf.log(1 - output + 1e-8)
    )
    return tf.negative(tf.reduce_mean(loss))


def loss_positions(targets: tf.Tensor, preds: tf.Tensor):
    """
    Args:
        targets: [#batch, h, w, (#anchor * [p, x, y, w, h])]
        preds: [#batch, h, w, (#anchor * [p, x, y, w, h])]
    """
    preds = tf.reshape(preds, (-1, 5))
    targets = tf.reshape(targets, (-1, 5))
    conf_preds = tf.sigmoid(preds[..., 0])
    conf_targets = targets[..., 0]
    mask = conf_targets > 0.9

    xy_preds = preds[..., 1:3]
    wh_preds = preds[..., 3:5]
    preds = tf.concat([xy_preds, wh_preds], axis=1)
    targets = targets[..., 1:]

    loss_conf = weighted_binary_cross_entropy(conf_preds, conf_targets, weights=[1, 10])
    loss_loc = tf.reduce_sum(
        tf.losses.mean_squared_error(
            targets, preds, reduction=tf.losses.Reduction.NONE
        ),
        axis=1,
    )
    loss_mean = tf.reduce_sum(loss_loc * tf.cast(mask, tf.float32)) / tf.reduce_sum(
        tf.cast(mask, tf.float32)
    )
    tf.summary.scalar("loss/position_loss/confidence", loss_conf)
    tf.summary.scalar("loss/position_loss/bounding_box", loss_mean)
    loss = loss_conf + 3 * loss_mean
    return loss


def calc_ocr_loss(lengths_pred, y_true, y_pred):
    """
    Args:
        y_true: [#batch, #max_boxes, #max_length]
        y_pred: [#batch, #max_boxes, #max_length, #labels]
        lengths_true: [#batch, #max_boxes]
        lengths_pred: [#batch, #max_boxes]
    """
    lengths_true = tf.reduce_sum(tf.cast(tf.not_equal(y_true, 0), tf.int32), axis=2)
    y_true_flatten = tf.reshape(y_true, [-1, tf.shape(y_true)[-1]])
    y_pred_flatten = tf.reshape(
        y_pred, [-1, tf.shape(y_pred)[-2], tf.shape(y_pred)[-1]]
    )
    lengths_true_flatten = tf.reshape(lengths_true, [-1])
    lengths_pred_flatten = tf.reshape(lengths_pred, [-1])
    indices = tf.where(
        tf.logical_and(
            tf.not_equal(lengths_true_flatten, 0),
            lengths_true_flatten < lengths_pred_flatten,
        )
    )
    # indices = tf.where(tf.not_equal(lengths_true_flatten, 0))
    y_true_flatten = tf.gather_nd(y_true_flatten, indices)
    y_pred_flatten = tf.gather_nd(y_pred_flatten, indices)
    lengths_true_flatten = tf.gather_nd(lengths_true_flatten, indices)
    lengths_pred_flatten = tf.gather_nd(lengths_pred_flatten, indices)
    loss = tf.keras.backend.ctc_batch_cost(
        y_true_flatten, y_pred_flatten, lengths_pred_flatten, lengths_true_flatten
    )
    return tf.reduce_mean(loss)


def decode_ocr(lengths, ocr_predictions):
    """
    Args:
        ocr_predictions: [#batch, #boxes, #max_time, #char]
    """
    shape = tf.shape(ocr_predictions)
    x = tf.reshape(ocr_predictions, [shape[0] * shape[1], shape[2], shape[3]])
    lengths = tf.reshape(lengths, [-1])
    decoded, probas = tf.keras.backend.ctc_decode(x, lengths, greedy=False)
    return tf.reshape(decoded, [shape[0], shape[1], -1])


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()

    sequence = Sequence("data/test", batch_size=8)
    model_fn = make_model_fn(sequence.anchor_boxes)
    config = tf.estimator.RunConfig(model_dir="checkpoint", save_checkpoints_secs=10)
    estimator = tf.estimator.Estimator(model_fn=model_fn, config=config)

    if not args.eval:
        train_spec = tf.estimator.TrainSpec(
            input_fn=Sequence("data/train").make_input_fn(batch_size=16),
            max_steps=1000000,
            hooks=[
                tf.train.StepCounterHook(every_n_steps=100, output_dir="checkpoint")
            ],
        )
        eval_spec = tf.estimator.EvalSpec(
            input_fn=Sequence("data/test").make_input_fn(batch_size=16)
        )

        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    else:
        from PIL import ImageDraw

        input_fn = Sequence("data/test").make_input_fn()
        for i, predictions in enumerate(estimator.predict(input_fn)):
            image = predictions["images"]
            box_confidences = predictions["box_confidences"]
            boxes = predictions["boxes"]
            ocrs = predictions["ocr"]
            img = Image.fromarray(image.astype(np.uint8))
            draw = ImageDraw.Draw(img)
            print("=========")
            for confidence, box, ocr in zip(box_confidences, boxes, ocrs):
                if box.sum() == 0:
                    break
                decoded = []
                for c in ocr:
                    if c == -1:
                        break
                    decoded.append(datagen.idx2char(c))
                print("confidence >", confidence)
                print("decoded >", "".join(decoded))
                draw.rectangle(list(box), outline="red")
            del draw
            print(img.save(f"out/{i}.png"))


if __name__ == "__main__":
    main()
