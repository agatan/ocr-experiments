import math
import tensorflow as tf
import numpy as np
import datagen
from group_norm import GroupNormalization

from icecream import ic


INPUT_SIZE = np.array([304, 192])
FEATURE_SIZE = np.array([76, 48])


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

class DataEncoder():
    def __init__(self):
        self.anchor_areas = [16.0*16.0, 32*32.0, 64*64.0]
        self.aspect_ratios = [2.0, 8.0, 16.0]
        self.anchor_wh = self._get_anchor_wh()

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

    def encode(self, boxes: np.ndarray, input_size: np.ndarray):
        '''
        Args:
            boxes: np.ndarray [#box, [xmin, ymin, xmax, ymax]]
            input_size: W, H
        Returns:
        loc_targets: np.ndarray [FH, FW, #anchor * [confidence, xcenter, ycenter, width, height]]
        '''
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
        return np.concatenate([masks, loc_targets], axis=3).reshape(fm_size[1], fm_size[0], 9 * 5)

    def reconstruct_bounding_boxes(self, outputs):
        '''
        Args:
            outputs: tensor [#batch, H, W, [preds, x, y, w, h]] (x, y in 0..1)
        Returns:
            boxes: tensor [#batch, H * W, [preds, x1, y1, x2, y2]]
        '''
        anchor_boxes = tf.convert_to_tensor(self._get_anchor_boxes(INPUT_SIZE), dtype=tf.float32)
        flatten = tf.reshape(outputs, (-1, FEATURE_SIZE[0] * FEATURE_SIZE[1] * 9, 5))
        out_xys = flatten[..., 1:3]
        out_whs = tf.sigmoid(flatten[..., 3:5])

        preds = tf.sigmoid(flatten[..., 0:1])
        xy = out_xys * anchor_boxes[:, 2:] + anchor_boxes[:, :2]
        wh = tf.exp(out_whs) * anchor_boxes[:, 2:]
        boxes = tf.concat([preds, xy - wh / 2, xy + wh / 2], axis=2)
        return boxes

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

        score = conf_preds # TODO: sigmoid
        ids = score > conf_thres
        ids = np.nonzero(ids)[0]
        return boxes[ids]

def generator(root: str, encoder: DataEncoder):
    import os
    import json
    from PIL import Image
    input_size = INPUT_SIZE
    fnames = []
    boxes = []
    texts = []
    lengths = []
    i = 0
    while True:
        f = os.path.join(root, f'{i}.json')
        i += 1
        if not os.path.isfile(f):
            break
        with open(f, 'r') as fp:
            info = json.load(fp)
        fnames.append(info['file'])
        bbs = np.zeros((20, 4))
        ts = np.zeros((20, 100), dtype=np.int32)
        lens = np.zeros(20, dtype=np.int32)
        for j, b in enumerate(info['boxes']):
            xmin = float(b['left'])
            ymin = float(b['top'])
            xmax = xmin + float(b['width'])
            ymax = ymin + float(b['height'])
            bbs[j] = [xmin, ymin, xmax, ymax]
            text = datagen.text2idx(b['text'])
            ts[j, :len(text)] = text
            lens[j] = len(text)
        boxes.append(np.array(bbs))
        texts.append(np.array(ts))
        lengths.append(lens)
    def g():
        for fname, bbs, ts, lens in zip(fnames, boxes, texts, lengths):
            img = Image.open(os.path.join(root, fname))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = img.resize((304, 192))
            img = np.array(img)
            loc_targets = encoder.encode(bbs, input_size)
            yield img, loc_targets, bbs, ts, lens
    return g


def _bottleneck(inputs, channels, strides):
    x = tf.keras.Sequential([
        tf.keras.layers.Conv2D(channels, kernel_size=1, use_bias=False),
        GroupNormalization(channels // 4),
        tf.keras.layers.Activation(tf.keras.activations.relu),
        tf.keras.layers.Conv2D(channels, kernel_size=3, strides=strides, use_bias=False, padding='same'),
        GroupNormalization(channels // 4),
        tf.keras.layers.Activation(tf.keras.activations.relu),
        tf.keras.layers.Conv2D(2 * channels, kernel_size=1, use_bias=False),
        GroupNormalization(channels // 4),
    ])(inputs)
    if strides != 1 or inputs.shape[-1] != x.shape[-1]:
        res = tf.keras.Sequential([
            tf.keras.layers.Conv2D(2 * channels, kernel_size=1, strides=strides, use_bias=False),
            GroupNormalization(channels // 4),
        ])(inputs)
        x = tf.keras.layers.Add()([x, res])
    return tf.keras.layers.Activation(tf.keras.activations.relu)(x)


def feature_extract(inputs):
    c1 = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, padding='same', use_bias=False),
        GroupNormalization(32),
        tf.keras.layers.Activation(tf.keras.activations.relu),
        tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same'),
    ])(inputs)
    c2 = _bottleneck(c1, 64, strides=1)
    c3 = _bottleneck(c2, 128, strides=2)
    c4 = _bottleneck(c3, 256, strides=2)
    c5 = _bottleneck(c4, 512, strides=2)
    p5 = tf.keras.layers.Conv2D(256, kernel_size=1, strides=1)(c5)
    x = tf.keras.layers.Conv2D(256, kernel_size=1, strides=1)(c4)
    p4 = tf.keras.layers.Add()([tf.keras.layers.UpSampling2D(size=(2, 2))(p5), x])
    p4 = tf.keras.layers.Conv2D(128, kernel_size=3, strides=1, padding='same')(p4)
    x = tf.keras.layers.Conv2D(128, kernel_size=1, strides=1)(c3)
    p3 = tf.keras.layers.Add()([tf.keras.layers.UpSampling2D(size=(2, 2))(p4), x])
    p3 = tf.keras.layers.Conv2D(128, kernel_size=3, strides=1, padding='same')(p3)
    return p3


def position_predict(feature_map):
    return tf.keras.layers.Conv2D(9 * 5, kernel_size=1, strides=1)(feature_map)


def weighted_binary_cross_entropy(output, target, weights):
    loss = weights[1] * (target * tf.log(output + 1e-8)) + weights[0] * ((1 - target) * tf.log(1 - output + 1e-8))
    return tf.negative(tf.reduce_mean(loss))


def loss_positions(loc_preds: tf.Tensor, loc_targets: tf.Tensor):
    '''
    Args:
        loc_preds: [#batch, h, w, (#anchor * [p, x, y, w, h])]
        loc_targets: [#batch, h, w, (#anchor * [p, x, y, w, h])]
    '''
    loc_preds = tf.reshape(loc_preds, (-1, 5))
    loc_targets = tf.reshape(loc_targets, (-1, 5))
    conf_preds = tf.sigmoid(loc_preds[..., 0])
    conf_targets = loc_targets[..., 0]
    mask = conf_targets > 0.9

    xy_preds = tf.sigmoid(loc_preds[..., 1:3])
    wh_preds = loc_preds[..., 3:5]
    loc_preds = tf.concat([xy_preds, wh_preds], axis=1)
    loc_targets = loc_targets[..., 1:]

    loss_conf = weighted_binary_cross_entropy(conf_preds, conf_targets, weights=[1, 10])
    loss_loc = tf.reduce_sum(tf.losses.mean_squared_error(loc_targets, loc_preds, reduction=tf.losses.Reduction.NONE), axis=1)
    loss_loc_mean = tf.reduce_sum(loss_loc * tf.cast(mask, tf.float32)) / tf.reduce_sum(tf.cast(mask, tf.float32))
    tf.summary.scalar('loss_conf', loss_conf)
    tf.summary.scalar('loss_location', loss_loc_mean)
    loss = loss_conf + 5 * loss_loc_mean
    return loss


def main():
    tf.enable_eager_execution()
    # return

    dataset = tf.data.Dataset.from_generator(generator('data/test', DataEncoder()), (tf.float32, tf.float32, tf.float32, tf.int32, tf.int32))
    dataset = dataset.map(lambda img, locs, bbs, texts, lengths: (tf.image.per_image_standardization(img), locs, bbs, texts, lengths))
    dataset = dataset.padded_batch(8, padded_shapes=([192, 304, 3], [FEATURE_SIZE[1], FEATURE_SIZE[0], 9 * 5], [None, 4], [None, 100], [None]))
    def mapper(img, locs, bbs, texts, lengths):
        return (img, dict(box=locs, bbs=bbs, texts=texts, lengths=lengths))
    dataset = dataset.map(mapper)

    ic(next(iter(dataset)))
    inputs = tf.keras.Input(shape=(192, 304, 3))
    feature_map = feature_extract(inputs)
    position = position_predict(feature_map)
    model = tf.keras.Model(inputs, position)
    optim = tf.train.AdamOptimizer()
    callback = tf.keras.callbacks.ModelCheckpoint("checkpoint")
    callback.set_model(model)
    model.compile(optimizer=optim, loss=loss_positions)
    model.fit_generator(dataset, callbacks=[callback])
    model.summary()
    print('Done')
    for (images, labels) in dataset:
        model.train_on_batch(images, labels['box'], call)
    print(model)


if __name__ == '__main__':
    main()
