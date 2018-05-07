import tensorflow as tf
from icecream import ic


def roi_pooling(image: tf.Tensor, boxes: tf.Tensor, height):
    cond = lambda i, out: i < tf.shape(image)[0]
    def body(i, out):
        bbs = boxes[i, ...]
        img = image[i, ...]

        base_widths = bbs[..., 2] - bbs[..., 0]
        base_heights = bbs[..., 3] - bbs[..., 1]
        aspects = base_widths / base_heights
        _, indices = tf.nn.top_k(aspects, k=tf.shape(aspects)[0])

        bbs = tf.gather(bbs, indices)
        base_widths = bbs[..., 2] - bbs[..., 0]
        base_heights = bbs[..., 3] - bbs[..., 1]

        result_bbs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, element_shape=[height, None, 3])
        inner_cond = lambda j, out, max_width: tf.logical_and(j < tf.shape(bbs)[0], tf.reduce_any(bbs[j] != 0))
        def inner_body(j, result_bbs, max_width):
            box = bbs[j, :]
            base_width = base_widths[j]
            base_height = base_heights[j]
            aspect = base_width / base_height
            width = tf.ceil(aspect * height)
            def then_branch():
                each_w = base_width / (width - 1)
                each_h = base_height / (height - 1)
                xx = tf.range(0, width, dtype=tf.float32) * each_w + box[0]
                yy = tf.range(0, height, dtype=tf.float32) * each_h + box[1]
                pooled = bilinear_interpolate(img, xx, yy)
                new_max_width = tf.maximum(max_width, width)
                ic(new_max_width)
                ic(tf.stack([height, new_max_width, 3]))
                # FIXME
                place = tf.zeros(tf.stack([height, new_max_width, 3]))
                ic(place)
                place[:, :width, :] = pooled
                return j + 1, result_bbs.write(j, place), new_max_width
            def else_branch():
                return j + 1, result_bbs, max_width
            return tf.cond(width > 0, then_branch, else_branch)
        _, result_bbs, _ = tf.while_loop(inner_cond, inner_body, [0, result_bbs, 0.0])
        i = tf.Print(i, [i])
        ic(result_bbs.stack())
        out = out.write(i, result_bbs.stack())
        return i + 1, out
    out = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, element_shape=[None, height, None, 3])
    _, out = tf.while_loop(cond, body, [0, out])
    return out.stack()


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
    ic(img_a)

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
    x = tf.placeholder(tf.float32, [None, 200, 300, 3])
    bbs = tf.placeholder(tf.float32, [None, None, 4])
    r = roi_pooling(x, bbs, 100)

    img = np.asarray(Image.open("test/0.png"))
    boxes = np.array([[0, 0, 100, 200], [100.1, 0, 300, 200]])
    with tf.Session() as sess:
        result = sess.run(r, {x: np.expand_dims(img, 0), bbs: np.expand_dims(boxes, 0)})
        ic(result)
    # img = Image.open("0.png")
    # img = transforms.ToTensor()(img)
    # img = torch.autograd.Variable(img)
    # r = RoIRotate(100)
    # img = img.unsqueeze(0)
    # bbs = torch.Tensor([[[0, 0, 100, 200], [100, 0, 300, 200]]])
    # boxes, masks = r.forward(img, bbs)
    # torchvision.utils.save_image(boxes.data[0], 'b.jpg')
#

main()
# x = tf.placeholder(tf.float32, [None, 32, 32, 3])
# boxes = tf.placeholder(tf.float32, [None, None, 4])
# r = roi_pooling(x, boxes, 2)
# with tf.Session() as sess:
#     import numpy as np
#     feed_boxes = np.zeros((2, 10, 4))
#     feed_boxes = np.random.rand(2, 10, 4) * 32
#     # feed_boxes[0, :, :] = 128
#     feed_boxes[1, 2:, :] = 0
#     result = sess.run(r, {x: np.ones([2, 32, 32, 3]), boxes: feed_boxes})
#     print(type(result))
#     print(result.shape)
