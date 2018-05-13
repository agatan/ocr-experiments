
# coding: utf-8

# In[1]:


import math
import tensorflow as tf
import numpy as np
from icecream import ic
import datagen


# In[2]:

INPUT_SIZE = np.array([300, 200])
FEATURE_SIZE = np.array([75, 50])


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
    input_size = np.array([300, 200])
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
            img = np.array(img)
            loc_targets = encoder.encode(bbs, input_size)
            yield img, loc_targets, bbs, ts, lens
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

_GROUP_NORMALIZATION_COUNT = 0

def group_normalization(x, G=32, eps=1e-5):
    global _GROUP_NORMALIZATION_COUNT
    with tf.variable_scope(f'group_normalization_{_GROUP_NORMALIZATION_COUNT}'):
        base_shape = x.shape
        _GROUP_NORMALIZATION_COUNT += 1
        x = tf.transpose(x, [0, 3, 1, 2])
        _, C, _, _ = x.get_shape().as_list() # C is static, but H, W, B is dynamic
        shape = tf.shape(x)
        # C = shape[1]
        H = shape[2]
        W = shape[3]
        G = tf.minimum(G, C)
        x = tf.reshape(x, tf.stack([-1, G, C // G, H, W]))
        mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)
        x = (x - mean) / tf.sqrt(var + eps)
        # per channel gamma and beta
        gamma = tf.get_variable('gamma', [C],
                                initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable('beta', [C],
                               initializer=tf.constant_initializer(0.0))
        gamma = tf.reshape(gamma, [1, C, 1, 1])
        beta = tf.reshape(beta, [1, C, 1, 1])

        output = tf.reshape(x, [-1, C, H, W]) * gamma + beta
        # tranpose: [bs, c, h, w, c] to [bs, h, w, c] following the paper
        output = tf.transpose(output, [0, 2, 3, 1])
        output.set_shape(base_shape)
        return output


def bottleneck(inputs: tf.Tensor, planes, strides=1, training=False):
    in_places = inputs.shape[-1]
    x = tf.layers.conv2d(inputs, planes, kernel_size=1, use_bias=False)
    x = group_normalization(x)
    x = tf.nn.relu(x)
    x = tf.layers.conv2d(x, planes, kernel_size=3, strides=strides, padding='same', use_bias=False)
    x = group_normalization(x)
    x = tf.nn.relu(x)
    x = tf.layers.conv2d(x, 2 * planes, kernel_size=1, use_bias=False)
    x = group_normalization(x)
    if strides != 1 or inputs.shape[-1] != x.shape[-1]:
        y = tf.layers.conv2d(inputs, x.shape[-1], kernel_size=1, strides=strides, use_bias=False)
        y = group_normalization(y)
        x += y
    return tf.nn.relu(x)

def upsampling_add(x, y):
    _, h, w, _ = y.shape
    return tf.image.resize_bilinear(x, size=(h, w)) + y

def feature_extract(inputs: tf.Tensor, training=False):
    with tf.name_scope("feature"):
        inputs = tf.identity(inputs, "inputs")
        x = tf.layers.conv2d(inputs, 64, kernel_size=3, strides=1, padding='same', use_bias=False)

    x = group_normalization(x)
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

def ocr_head(features: tf.Tensor, training=False):
    '''
    Args:
        features: tensor [#batch, 8, W, C]
    '''
    def block(x, channels):
        x = tf.layers.conv2d(x, channels, kernel_size=3, strides=1, padding='same')
        x = tf.nn.relu(group_normalization(x))
        x = tf.layers.max_pooling2d(x, (2, 1), strides=(2, 1))
        return x

    import datagen

    with tf.name_scope("ocr"):
        x = features
        for c in [128, 256, len(datagen._charset()) + 1]:
            x = block(x, c)
        return tf.squeeze(x, axis=1)

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


def roi_pooling(image: tf.Tensor, boxes: tf.Tensor, height):
    base_widths = boxes[:, 2] - boxes[:, 0]
    base_heights = boxes[:, 3] - boxes[:, 1]
    aspects = base_widths / base_heights
    widths = tf.ceil(aspects * float(height))
    max_width = tf.cast(tf.reduce_max(widths), dtype=tf.int32)

    def mapper(box):
        cond = tf.reduce_any(tf.not_equal(box, 0))
        def then_branch():
            base_width = box[2] - box[0]
            base_height = box[3] - box[1]
            aspect = base_width / base_height
            width = tf.ceil(aspect * float(height))
            map_w = base_width / (width - 1)
            map_h = base_height / (height - 1)
            xx = tf.range(0, width, dtype=tf.float32) * map_w + box[0]
            yy = tf.range(0, height, dtype=tf.float32) * map_h + box[1]
            pooled = bilinear_interpolate(image, xx, yy)
            return tf.pad(
                pooled, [[0, 0], [0, max_width - tf.cast(width, tf.int32)], [0, 0]])
        def else_branch():
            return tf.zeros((height, max_width, tf.shape(image)[-1]))
        return tf.cond(cond, then_branch, else_branch)

    results = tf.map_fn(mapper, boxes)
    results.set_shape([None, height, None, image.shape[-1]])
    return results


def bilinear_interpolate(img: tf.Tensor, x: tf.Tensor, y: tf.Tensor):
    '''
    Args:
        img: [H, W, C]
        x: [-1]
        y: [-1]
    '''
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

# In[30]:

def model_fn(features, labels, mode):
    training = mode == tf.estimator.ModeKeys.TRAIN
    fm = feature_extract(features, training=training)
    loc_preds = position_prediction_head(fm)

    if mode == tf.estimator.ModeKeys.PREDICT:
        encoder = DataEncoder()
        boxes = encoder.reconstruct_bounding_boxes(loc_preds)
        def nms_fn(boxes):
            indices = tf.image.non_max_suppression(boxes[:, 1:], boxes[:, 0], 64)
            return tf.gather(boxes, indices)
        boxes = tf.map_fn(nms_fn, boxes, dtype=tf.float32)
        predictions = dict(boxes=boxes)
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    with tf.name_scope("myloss"):
        loss = loss_positions(loc_preds, labels['box'])

    bbs = labels['bbs']
    texts = labels['texts']
    lengths = labels['lengths']
    def mapper(i):
        feature, boxes, targets, lens = fm[i], bbs[i], texts[i], lengths[i]
        pooled = roi_pooling(feature, boxes, 8)
        ocr_results = tf.nn.softmax(ocr_head(pooled))
        targets = tf.one_hot(targets, depth=len(datagen.CHARSET), axis=1)
        indices = tf.where(tf.not_equal(targets, 0))
        values = tf.gather_nd(targets, indices)
        targets = tf.SparseTensor(indices, values, tf.shape(targets, out_type=tf.int64))
        loss = tf.reduce_mean(tf.nn.ctc_loss(tf.cast(targets, dtype=tf.int32), ocr_results, lens, time_major=False))
        return loss

    loss_mean = tf.reduce_sum(tf.map_fn(mapper, tf.range(tf.shape(fm)[0]), dtype=tf.float32))
    loss = loss_mean

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=dict())

    optimizer = tf.train.AdamOptimizer()
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

def input_fn(root):
    ic('input_fn')
    g = generator(root, DataEncoder())
    dataset = tf.data.Dataset.from_generator(g, (tf.float32, tf.float32, tf.float32, tf.int32, tf.int32))
    dataset = dataset.map(lambda img, locs, bbs, texts, lengths: (tf.image.per_image_standardization(img), locs, bbs, texts, lengths))
    dataset = dataset.padded_batch(1, padded_shapes=([200, 300, 3], [FEATURE_SIZE[1], FEATURE_SIZE[0], 9 * 5], [None, 4], [None, 100], [None]))
    def mapper(img, locs, bbs, texts, lengths):
        return (img, dict(box=locs, bbs=bbs, texts=texts, lengths=lengths))
    dataset = dataset.map(mapper)

    return dataset

def main():
    from tensorflow.python import debug as tf_debug

    # tf.enable_eager_execution()
    # for x in input_fn('train'):
    #     ic(x)
    #     import sys; sys.exit(0)

    # Create a LocalCLIDebugHook and use it as a monitor when calling fit().
    hooks = []
    # hooks = [tf_debug.LocalCLIDebugHook()]
    hooks = [tf_debug.TensorBoardDebugHook("localhost:2333")]
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
            filenames = tf.constant([f'test/{n}.png' for n in range(0, 10)])
            return tf.data.Dataset.from_tensor_slices(filenames).map(lambda f: tf.image.decode_png(tf.read_file(f), channels=3)).map(lambda img: tf.image.resize_images(img, [200, 300])).map(lambda img: (tf.image.per_image_standardization(img), dict())).batch(4)

        for out in estimator.predict(eval_input_fn):
            img = Image.open(f"test/{n}.png")
            draw = ImageDraw.Draw(img)
            for box in out['boxes']:
                draw.rectangle(list(box[1:]), outline='red')
            img.show()
            print(out)
    else:
        for epoch in range(100):
            print(f'Epoch {epoch + 1}:')
            estimator.train(train_input_fn, hooks=hooks)
            estimator.evaluate(test_input_fn, hooks=hooks)

main()

