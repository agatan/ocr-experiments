import tensorflow as tf


def roi_pooling(image: tf.Tensor, boxes: tf.Tensor, height):
    base_widths = boxes[:, 2] - boxes[:, 0]
    base_heights = boxes[:, 3] - boxes[:, 1]
    aspects = base_widths / base_heights
    widths = tf.ceil(aspects * float(height))
    max_width = tf.cast(tf.reduce_max(widths), dtype=tf.int32)

    def mapper(box):
        base_width = box[2] - box[0]
        base_height = box[3] - box[1]
        aspect = base_width / base_height
        width = tf.ceil(aspect * float(height))
        map_w = base_width / (width - 1)
        map_h = base_height / (height - 1)
        xx = tf.range(0, width, dtype=tf.float32) * map_w + box[0]
        yy = tf.range(0, height, dtype=tf.float32) * map_h + box[1]
        pooled = bilinear_interpolate(image, xx, yy)
        padded = tf.pad(
            pooled, [[0, 0], [0, max_width - tf.cast(width, tf.int32)], [0, 0]])
        return padded

    return tf.map_fn(mapper, boxes)


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


def main():
    from PIL import Image
    import numpy as np
    x = tf.placeholder(tf.float32, [200, 300, 3])
    bbs = tf.placeholder(tf.float32, [None, 4])
    r = roi_pooling(x, bbs, 100)

    img = np.asarray(Image.open("test/0.png"))
    boxes = np.array([[0, 0, 100, 200], [100.1, 0, 300, 200]])
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        _, result = sess.run([init, r], {x: img, bbs: boxes})
        new_image = np.vstack(result).astype(np.uint8)
        new_image = Image.fromarray(new_image)
        new_image.show()


main()
