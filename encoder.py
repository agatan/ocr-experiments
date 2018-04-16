import math
import torch


def meshgrid(x, y):
    a = torch.arange(0, x)
    b = torch.arange(0, y)
    xx = a.repeat(y).view(-1, 1)
    yy = b.view(-1, 1).repeat(1, x).view(-1, 1)
    return torch.cat([xx, yy], 1)


def xywh2xyxy(boxes):
    xy = boxes[..., :2]
    wh = boxes[..., 2:]
    return torch.cat([xy - wh / 2, xy + wh / 2], -1)


def xyxy2xywh(boxes):
    xymin = boxes[:, :2]
    xymax = boxes[:, 2:]
    return torch.cat([(xymin + xymax) / 2, xymax - xymin + 1], 1)


def box_iou(box1, box2):
    lt = torch.max(box1[..., None, :2], box2[:, :2])  # N, M, 2
    rb = torch.min(box1[..., None, 2:], box2[:, 2:])  # N, M, 2

    wh = (rb - lt + 1).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]  # N, M
    area1 = (box1[..., 2] - box1[..., 0] + 1) * (box1[..., 3] - box1[..., 1] + 1)
    area2 = (box2[..., 2] - box2[..., 0] + 1) * (box2[..., 3] - box2[..., 1] + 1)
    iou = inter / (area1[:, None] + area2 - inter)
    return iou


def box_nms(bboxes, scores, thres=0.5):
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    _, order = scores.sort(0, descending=True)

    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)
        if order.numel() == 1:
            break
        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])

        w = (xx2 - xx1 + 1).clamp(min=0)
        h = (yy2 - yy1 + 1).clamp(min=0)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        ids = (ovr <= thres).nonzero().squeeze()
        if ids.numel() == 0:
            break
        order = order[ids+1]
    return torch.LongTensor(keep)


class DataEncoder():
    def __init__(self):
        self.anchor_areas = [32*32., 64*64., 128 * 128.]  # p3 -> p7
        self.aspect_ratios = [2/1., 4/1., 8/1.]
        self.anchor_wh = self._get_anchor_wh()

    def _get_anchor_wh(self):
        anchor_wh = []
        for s in self.anchor_areas:
            for ar in self.aspect_ratios:
                h = math.sqrt(s / ar)
                w = ar * h
                anchor_wh.append([w, h])
        return torch.Tensor(anchor_wh).view(9, 2)

    def _get_anchor_boxes(self, input_size):
        fm_size = (input_size / pow(2, 2)).ceil()

        grid_size = input_size / fm_size
        fm_w, fm_h = int(fm_size[0]), int(fm_size[1])
        xy = meshgrid(fm_w, fm_h) + 0.5
        xy = (xy * grid_size).view(fm_h, fm_w, 1, 2).expand(fm_h,
                                                            fm_w, 9, 2)  # (fm_h, fm_w, #anchor, (x, y)
        wh = self.anchor_wh.view(1, 1, 9, 2).expand(fm_h, fm_w, 9, 2)
        box = torch.cat([xy, wh], 3)  # [x, y, w, h]
        return box.view(-1, 4)

    def encode(self, boxes, input_size):
        '''
        Args:
            boxes: tensor [#box, [xmin, ymin, xmax, ymax]]
            input_size: (W, H)
        Returns:
            loc_targets: tensor [#anchor(9) * [confidence, xcenter, ycenter, width, height], FH, FW]
        '''
        fm_size = [math.ceil(i / pow(2, 2)) for i in input_size]
        input_size = torch.Tensor(input_size)
        anchor_boxes = self._get_anchor_boxes(input_size)

        ious = box_iou(xywh2xyxy(anchor_boxes), boxes)
        boxes = xyxy2xywh(boxes)

        max_ious, max_ids = ious.max(1)
        boxes = boxes[max_ids]

        loc_xy = (boxes[:, :2] - anchor_boxes[:, :2]) / anchor_boxes[:, 2:]
        loc_wh = torch.log(boxes[:, 2:] / anchor_boxes[:, 2:])
        loc_targets = torch.cat([loc_xy, loc_wh], 1)

        masks = torch.ones(max_ids.size())
        masks[max_ious < 0.7] = 0
        masks[(max_ious > 0.3) & (max_ious < 0.7)] = -1

        loc_targets = loc_targets.contiguous().view(fm_size[1], fm_size[0], 9, 4)
        masks = masks.contiguous().view(fm_size[1], fm_size[0], 9, 1)
        return torch.cat((masks, loc_targets), 3).view(fm_size[1], fm_size[0], 9 * 5).permute(2, 0, 1)

    def decode(self, loc_preds, conf_preds, input_size):
        input_size = torch.Tensor(input_size)
        anchor_boxes = self._get_anchor_boxes(input_size)

        loc_xy = loc_preds[:, :2]
        loc_wh = loc_preds[:, 2:]

        xy = loc_xy * anchor_boxes[:, 2:] + anchor_boxes[:, :2]
        wh = loc_wh.exp() * anchor_boxes[:, 2:]
        boxes = torch.cat([xy - wh / 2, xy + wh / 2], 1)

        score = conf_preds.sigmoid().squeeze(1)
        ids = score > 0.5
        ids = ids.nonzero().squeeze()
        if len(ids) == 0:
            return None
        keep = box_nms(boxes[ids], score[ids])
        return boxes[ids][keep]
