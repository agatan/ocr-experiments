import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw

from data import CharDictionary, reconstruct_boxes
from backbone import ResNet50Backbone
from model import Recognition, Detection
from roirotate import roirotate


# Original author: Francisco Massa:
# https://github.com/fmassa/object-detection.torch
# Ported to PyTorch by Max deGroot (02/01/2017)
def nms(boxes, scores, overlap=0.5, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """

    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w*h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep, count


chardict = CharDictionary("0123456789")
backbone = ResNet50Backbone()
recognition = Recognition(chardict.vocab)
detection = Detection()

state = torch.load("./checkpoint/best.pth.tar", map_location=lambda storage, log: storage)
backbone.load_state_dict(state['backbone'])
recognition.load_state_dict(state['recognition'])
detection.load_state_dict(state['detection'])
backbone.eval()
recognition.eval()
detection.eval()

image = Image.open("./data/test/1.png")
image_tensor = transforms.ToTensor()(image).unsqueeze(0)
with torch.no_grad():
    feature_map = backbone(image_tensor)
    detections_pred = detection(feature_map)

    for detection_pred in detections_pred:
        recons = reconstruct_boxes(detection_pred[1:, :, :])
        scores = detection_pred[0, :, :].view(-1)
        recons = recons.view(4, -1).transpose(0, 1)
        keep, count = nms(recons, scores, top_k=200)
        scores = torch.sigmoid_(scores[keep[:count]])
        recons = recons[keep[:count]]
        recons = recons[(recons[:, 2] - recons[:, 0] > 1) & (recons[:, 3] - recons[:, 1] > 1)]
        boxes, masks = roirotate(feature_map[0], recons, height=8)
        recognized = recognition(boxes)
        argmax = torch.argmax(recognized, dim=2)
        copy = image.copy().convert("RGBA")
        draw = ImageDraw.Draw(image)

        for xmin, ymin, xmax, ymax in recons * 4:
            draw.rectangle(((xmin, ymin), (xmax, ymax)), outline=(255, 0, 0), width=2)
        for i, (xmin, ymin, xmax, ymax) in enumerate(recons * 4):
            last = None
            decoded = []
            for idx in argmax[i]:
                if last is not None and last == idx:
                    continue
                last = idx
                if idx == 0:
                    continue
                c = chardict.idx2char(idx)
                decoded.append(c)
            print(''.join(decoded))
            draw.text((xmin - 5, ymin - 5), text=''.join(decoded), fill=(0, 255, 0))
        image = image.convert("RGBA")
        image.show()
        copy.putalpha(128)
        image.putalpha(128)
        Image.alpha_composite(image, copy).show()
