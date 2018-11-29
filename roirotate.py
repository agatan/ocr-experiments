import torch
import torch.nn.functional as F
from icecream import ic



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
    max_width = torch.max(new_widths).item()
    max_boxes = boxes.size()[1]

    batch_size, c_img, h_img, w_img = image.size()
    results = []
    masks = []
    for i in range(image.size(0)):
        mask = torch.zeros(max_boxes, max_width).byte()
        img = image[i]
        bbs = boxes[i]
        xxs = torch.ones((bbs.size(0), height, max_width)) * -2.0
        yys = torch.full((bbs.size(0), height, max_width), -2.0)
        for j in range(bbs.size(0)):
            box = bbs[j]
            base_width = base_widths[i, j].item()
            base_height = base_heights[i, j].item()
            if bool(base_width != 0) and bool(base_height != 0):
                width = new_widths[i, j].item()
                each_w = base_width / (width - 1)
                each_h = base_height / (height - 1)
                xx = torch.arange(
                    0, width, dtype=torch.float32) * each_w + box[0]
                xx = xx.view(1, -1).repeat(height, 1).view(height, width)
                xxs[j, :, :width] = (xx - w_img / 2) / (w_img / 2)
                yy = torch.arange(0, height, dtype=torch.float32) * each_h + box[1]
                yy = yy.view(-1, 1).repeat(1, width).view(height, width)
                yys[j, :, :width] = (yy - h_img / 2) / (h_img / 2)
                mask[j, :width] = 1
        results.append(F.grid_sample(img.repeat(bbs.size(0), 1, 1, 1), torch.stack([xxs, yys], -1)))
        masks.append(mask)

    result = torch.stack(results, 0)
    mask = torch.stack(masks, 0)
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
    img = Image.open("./data/train/0.png")
    img = transforms.ToTensor()(img).requires_grad_(True)
    bbs = torch.Tensor([[10, 57, 74, 78], [100.1, 0, 300, 200]]).requires_grad_(True)
    boxes, masks = roirotate(img, bbs, height=100, vertical=False)
    print(boxes.size(), boxes.dtype)
    boxes.mean().backward()
    torchvision.utils.save_image(boxes.data, 'a.jpg')

    img = Image.open("./data/train/0.png")
    img = transforms.ToTensor()(img)
    img = img.unsqueeze(0).requires_grad_(True)
    bbs = torch.Tensor([[[0, 0, 100, 200], [100, 0, 300, 200], [0, 0, 0, 0]]]).requires_grad_(True)
    boxes, masks = roirotate(img, bbs, height=100, vertical=False)
    print(boxes.size())
    boxes.mean().backward()
    torchvision.utils.save_image(boxes.data[0], 'b.jpg')


if __name__ == "__main__":
    main()
