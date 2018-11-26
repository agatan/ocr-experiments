import torchvision.transforms as transforms
import torchvision
from PIL import Image
import torch
import torch.nn as nn


class RoIRotate(nn.Module):
    def __init__(self, height, vertical=False):
        super().__init__()
        self.height = height
        self.vertical = vertical

    def forward(self, image, boxes):
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

        if self.vertical:
            image = torch.transpose(image, 2, 3)
            new_boxes = torch.zeros_like(boxes)
            new_boxes[..., 0] = boxes[..., 1]
            new_boxes[..., 1] = boxes[..., 0]
            new_boxes[..., 2] = boxes[..., 3]
            new_boxes[..., 3] = boxes[..., 2]
            boxes = new_boxes

        batch_size = image.size(0)
        base_widths = boxes[..., 2] - boxes[..., 0]
        base_heights = boxes[..., 3] - boxes[..., 1]
        aspects = base_widths.type(torch.float32) / base_heights.type(torch.float32)
        new_widths = aspects * self.height
        new_widths[torch.isnan(new_widths)] = 0
        new_widths = new_widths.type(torch.int32)
        max_width = new_widths.max()
        max_boxes = boxes.size()[1]

        batch_result_boxes = []
        for i in range(image.size()[0]):
            result_bbs = []
            img = image[i]
            bbs = boxes[i]
            c_img, h_img, w_img = img.size()
            for j in range(bbs.size()[0]):
                box = bbs[j]
                base_width = base_widths[i, j]
                base_height = base_heights[i, j]
                if base_width == 0 and base_height == 0:
                    result_bbs.append(torch.zeros((img.size()[0], self.height, 0), dtype=torch.float32))
                    continue
                width = new_widths[i, j]
                each_w = base_width / (width - 1)
                each_h = base_height / (self.height - 1)
                xx = torch.arange(
                    0, width, dtype=torch.float32) * each_w + box[0]
                xx = xx.view(1, -1).repeat(self.height,
                                           1).view(self.height, width)
                yy = torch.arange(
                    0, self.height, dtype=torch.float32) * each_h + box[1]
                yy = yy.view(-1, 1).repeat(1, width).view(self.height, width)
                result = bilinear_interpolate_torch(img, xx, yy)
                result_bbs.append(result)
            batch_result_boxes.append(result_bbs)

        result = torch.zeros(batch_size, max_boxes, c_img,
                             self.height, max_width)
        mask = torch.zeros(batch_size, max_boxes, max_width).byte()
        for i, boxes in enumerate(batch_result_boxes):
            for j, box in enumerate(boxes):
                _, _, w = box.size()
                mask[i, j, :w] = 1
                result[i, j, :, :, :w] = box

        if self.vertical:
            result = torch.transpose(result, 3, 4)
        if expand:
            result.squeeze_(0)
            mask.squeeze_(0)

        return result, mask


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


def main():
    from PIL import Image
    import torchvision
    import torchvision.transforms as transforms
    img = Image.open("./out/0.png")
    img = transforms.ToTensor()(img).requires_grad_(True)
    r = RoIRotate(100, vertical=True)

    bbs = torch.Tensor([[0, 0, 100, 200], [100.1, 0, 300, 200]]).requires_grad_(True)
    boxes, masks = r.forward(img, bbs)
    boxes.mean().backward()
    print(bbs.grad, img.grad)
    torchvision.utils.save_image(boxes.data, 'a.jpg')

    img = Image.open("./out/1.png")
    img = transforms.ToTensor()(img)
    r = RoIRotate(100)
    img = img.unsqueeze(0).requires_grad_(True)
    bbs = torch.Tensor([[[0, 0, 100, 200], [100, 0, 300, 200], [0, 0, 0, 0]]]).requires_grad_(True)
    boxes, masks = r.forward(img, bbs)
    print(boxes.size())
    boxes.mean().backward()
    print(bbs.grad, img.grad)
    torchvision.utils.save_image(boxes.data[0], 'b.jpg')
