
# coding: utf-8

# In[1]:


import math
import tensorflow as tf
import numpy as np


# In[2]:


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
    input_size = np.array([300, 200])
    fnames = []
    boxes = []
    texts = []
    i = 0
    while True:
        f = os.path.join(root, f'{i}.json')
        i += 1
        if not os.path.isfile(f):
            break
        with open(f, 'r') as fp:
            info = json.load(fp)
        fnames.append(info['file'])
        bbs = []
        ts = []
        for b in info['boxes']:
            xmin = float(b['left'])
            ymin = float(b['top'])
            xmax = xmin + float(b['width'])
            ymax = ymin + float(b['height'])
            bbs.append([xmin, ymin, xmax, ymax])
            ts.append(bytes(b['text'], 'utf8'))
        boxes.append(np.array(bbs))
        texts.append(np.array(ts))
    def g():
        for fname, bbs, ts in zip(fnames, boxes, texts):
            img = Image.open(os.path.join(root, fname))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = np.array(img)
            loc_targets = encoder.encode(bbs, input_size)
            yield img, loc_targets, bbs, ts
    return g

def test():
    g = generator('test', DataEncoder())
    dataset = tf.data.Dataset.from_generator(g, (tf.float32, tf.float32, tf.float32, tf.string))
    dataset = dataset.map(lambda img, locs, bbs, texts: (tf.image.per_image_standardization(img), locs, bbs, texts))
    dataset = dataset.padded_batch(10, padded_shapes=([None, None, None], [None, None, None], [None, 5], [None]))
    import tensorflow.contrib.eager as tfe
    from PIL import Image, ImageDraw
    img, loc, bbs, texts = next(tfe.Iterator(dataset))
    img = img[0]
    loc = loc[0]
    decoded_boxes = DataEncoder().decode(loc.numpy(), np.array([300, 200]))
    img = Image.fromarray(np.uint8(img.numpy()))
    draw = ImageDraw.Draw(img)
    for box in decoded_boxes:
        draw.rectangle(list(box), outline='red')
    img.show()


# test()


# In[3]:


def bottleneck(inputs: tf.Tensor, planes, strides=1, training=False):
    in_places = inputs.shape[-1]
    x = tf.layers.conv2d(inputs, planes, kernel_size=1, use_bias=False)
    x = tf.layers.batch_normalization(x, training=training)
    x = tf.nn.relu(x)
    x = tf.layers.conv2d(x, planes, kernel_size=3, strides=strides, padding='same', use_bias=False)
    x = tf.layers.batch_normalization(x, training=training)
    x = tf.nn.relu(x)
    x = tf.layers.conv2d(x, 2 * planes, kernel_size=1, use_bias=False)
    x = tf.layers.batch_normalization(x, training=training)
    if strides != 1 or inputs.shape[-1] != x.shape[-1]:
        y = tf.layers.conv2d(inputs, x.shape[-1], kernel_size=1, strides=strides, use_bias=False)
        y = tf.layers.batch_normalization(y, training=training)
        x += y
    return tf.nn.relu(x)

def upsampling_add(x, y):
    _, h, w, _ = y.shape
    return tf.image.resize_bilinear(x, size=(h, w)) + y

def feature_extract(inputs: tf.Tensor, training=False):
    with tf.name_scope("feature"):
        inputs = tf.identity(inputs, "inputs")
        x = tf.layers.conv2d(inputs, 64, kernel_size=3, strides=1, padding='same', use_bias=False)

    x = tf.layers.batch_normalization(x, training=training)
    c1 = tf.layers.max_pooling2d(tf.nn.relu(x), pool_size=3, strides=2, padding='same')
    c2 = bottleneck(c1, 64, strides=1, training=training)
    c3 = bottleneck(c2, 128, strides=2, training=training)
    c4 = bottleneck(c3, 256, strides=2, training=training)
    c5 = bottleneck(c4, 512, strides=2, training=training)
    p5 = tf.layers.conv2d(c5, 256, kernel_size=1, strides=1)
    p4 = upsampling_add(p5, tf.layers.conv2d(c4, 256, kernel_size=1, strides=1))
    p4 = tf.layers.conv2d(p4, 128, kernel_size=3, strides=1, padding='same')
    p3 = upsampling_add(p4, tf.layers.conv2d(c3, 128, kernel_size=1, strides=1))
    p3 = tf.layers.conv2d(p3, 128, kernel_size=3, strides=1, padding='same')
    return p3

def position_prediction_head(fm: tf.Tensor):
    with tf.name_scope("position_prediction_head"):
        return tf.layers.conv2d(fm, 9 * 5, kernel_size=1, strides=1)


# In[28]:


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
    loss = loss_conf + 5 * loss_loc_mean
    return loss


# In[30]:


def model_fn(features, labels, mode):
    training = mode == tf.estimator.ModeKeys.TRAIN
    fm = feature_extract(features, training=training)
    loc_preds = position_prediction_head(fm)
    if mode == tf.estimator.ModeKeys.PREDICT:
        loc_preds = tf.reshape(loc_preds, (-1, 9, 5))
        confidences = tf.sigmoid(loc_preds[..., 0:1])
        loc_xy = tf.sigmoid(loc_preds[..., 1:3])
        loc_wh = loc_preds[..., 3:5]
        loc_preds = tf.concat([confidences, loc_xy, loc_wh], axis=-1)
        loc_preds = tf.reshape(loc_preds, (-1, 50 * 75 * 9, 5))
        predictions = dict(boxes=loc_preds)
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    with tf.name_scope("myloss"):
        loss = loss_positions(loc_preds, labels['box'])
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=dict())

    optimizer = tf.train.AdamOptimizer()
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

def input_fn(root):
    g = generator(root, DataEncoder())
    dataset = tf.data.Dataset.from_generator(g, (tf.float32, tf.float32, tf.float32, tf.string))
    def mapper(img, locs, bbs, texts):
        return (img, dict(box=locs, bbs=bbs, texts=texts))
    dataset = dataset.map(lambda img, locs, bbs, texts: (tf.image.per_image_standardization(img), locs, bbs, texts))
    dataset = dataset.padded_batch(32, padded_shapes=([200, 300, 3], [None, None, None], [None, 5], [None]))
    dataset = dataset.map(mapper)

    return dataset

def main():
    from tensorflow.python import debug as tf_debug

    # Create a LocalCLIDebugHook and use it as a monitor when calling fit().
    # hooks = [tf_debug.LocalCLIDebugHook()]
    # hooks = [tf_debug.TensorBoardDebugHook("localhost:2333")]
    hooks = []
    config = tf.estimator.RunConfig(
        model_dir='/tmp/checkpoint',
        save_checkpoints_secs=10,
    )
    estimator = tf.estimator.Estimator(model_fn=model_fn, config=config)

    def train_input_fn():
        return input_fn('train')

    def test_input_fn():
        return input_fn('test')

    EVAL = False
    if EVAL:
        from PIL import Image, ImageDraw
        n = np.random.randint(0, 99)
        def eval_input_fn():
            imgs = np.array(Image.open(f"test/{n}.png").convert("RGB"))
            return tf.data.Dataset.from_tensors(imgs).map(lambda img: (tf.image.per_image_standardization(img), dict())).batch(1)

        for out in estimator.predict(eval_input_fn):
            decoded = DataEncoder().decode(out['boxes'], np.array([300, 200]))
            img = Image.open(f"test/{n}.png")
            draw = ImageDraw.Draw(img)
            for box in decoded:
                draw.rectangle(list(box), outline='red')
            img.show()
            print(out)
    else:
        for epoch in range(100):
            print(f'Epoch {epoch + 1}:')
            estimator.train(train_input_fn, hooks=hooks)
            estimator.evaluate(test_input_fn, hooks=hooks)

main()

