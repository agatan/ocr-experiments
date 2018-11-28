import torch

def bilinear_interpolate_torch(im, x, y):
    x0 = torch.floor(x).long()
    x1 = x0 + 1

    y0 = torch.floor(y).long()
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, im.shape[2]-1)
    x1 = torch.clamp(x1, 0, im.shape[2]-1)
    y0 = torch.clamp(y0, 0, im.shape[1]-1)
    y1 = torch.clamp(y1, 0, im.shape[1]-1)

    im_a = im[:, y0, x0]
    im_b = im[:, y1, x0]
    im_c = im[:, y0, x1]
    im_d = im[:, y1, x1]

    wa = (x1.float()-x) * (y1.float()-y)
    wb = (x1.float()-x) * (y-y0.float())
    wc = (x-x0.float()) * (y1.float()-y)
    wd = (x-x0.float()) * (y-y0.float())

    return im_a * wa + im_b * wb + im_c * wc + im_d * wd


def roirotate(image, boxes, height, vertical=False):
    # type: (Tensor, Tensor, int, bool) -> Tuple[Tensor, Tensor]
    '''
    Args:
        image: tensor [#batch, Channel, H, W] or [Channel, H, W]
        boxes: tensor [#batch, #boxes, 5 (left, top, right, bottom, theta)] or [#boxes, 5]
    '''
    expand = False
    if len(image.size()) == 3:
        image = image.unsqueeze(0)
        boxes = boxes.unsqueeze(0)
        expand = True
    else:
        assert image.size(0) == boxes.size(0)

    if bool(vertical):
        image = torch.transpose(image, 2, 3)
        new_boxes = torch.zeros_like(boxes)
        new_boxes[:, :, 0] = boxes[:, :, 1]
        new_boxes[:, :, 1] = boxes[:, :, 0]
        new_boxes[:, :, 2] = boxes[:, :, 3]
        new_boxes[:, :, 3] = boxes[:, :, 2]
        boxes = new_boxes

    batch_size = image.size(0)
    base_widths = boxes[:, :, 2] - boxes[:, :, 0]
    base_heights = boxes[:, :, 3] - boxes[:, :, 1]
    aspects = base_widths.float() / base_heights.float()
    new_widths = aspects * height
    new_widths[new_widths != new_widths] = 0
    new_widths = new_widths.round_().int()
    max_width = torch.max(new_widths)
    max_boxes = boxes.size()[1]

    c_img = image.size(1)
    result = torch.zeros(batch_size, max_boxes, c_img,
                         height, max_width)
    mask = torch.zeros(batch_size, max_boxes, max_width).byte()
    for i in range(image.size(0)):
        img = image[i]
        bbs = boxes[i]
        c_img, h_img, w_img = img.size()
        for j in range(bbs.size(0)):
            box = bbs[j]
            base_width = base_widths[i, j]
            base_height = base_heights[i, j]
            if bool(base_width != 0) and bool(base_height != 0):
                width = new_widths[i, j]
                each_w = base_width / (width - 1)
                each_h = base_height / (height - 1)
                xx = torch.arange(
                    0, width, dtype=torch.float32) * each_w + box[0]
                xx = xx.view(1, -1).repeat(height,
                                           1).view(height, width)
                yy = torch.arange(
                    0, height, dtype=torch.float32) * each_h + box[1]
                yy = yy.view(-1, 1).repeat(1, width).view(height, width)
                result_box = bilinear_interpolate_torch(img, xx.to(img.device), yy.to(img.device))
                w = result_box.size(2)
                result[i, j, :, :, :w] = result_box
                mask[i, j, :w] = 1

    if vertical:
        result = torch.transpose(result, 3, 4)
    if expand:
        result.squeeze_(0)
        mask.squeeze_(0)

    return result.to(image.device), mask.to(image.device)


def main():
    from PIL import Image
    import torchvision
    import torchvision.transforms as transforms
    img = Image.open("./out/0.png")
    img = transforms.ToTensor()(img).requires_grad_(True)
    bbs = torch.Tensor([[0, 0, 100, 200], [100.1, 0, 300, 200]]).requires_grad_(True)
    boxes, masks = roirotate(img, bbs, height=100, vertical=True)
    boxes.mean().backward()
    print(bbs.grad, img.grad)
    torchvision.utils.save_image(boxes.data, 'a.jpg')

    img = Image.open("./out/1.png")
    img = transforms.ToTensor()(img)
    img = img.unsqueeze(0).requires_grad_(True)
    bbs = torch.Tensor([[[0, 0, 100, 200], [100, 0, 300, 200], [0, 0, 0, 0]]]).requires_grad_(True)
    boxes, masks = roirotate(img, bbs, height=100, vertical=False)
    print(boxes.size())
    boxes.mean().backward()
    print(bbs.grad, img.grad)
    torchvision.utils.save_image(boxes.data[0], 'b.jpg')


if __name__ == "__main__":
    main()
