import torch.nn as nn
import torch.nn.functional as F


class Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, loc_preds, loc_targets, mask):
        '''
        Args:
            loc_preds: [batch_size, #anchors, 4]
            loc_targets: [batch_size, #anchors, 4]
            mask: [batch_size, #anchors]
        '''
        batch_size, n_boxes, _ = loc_targets.size()
        n_obj = mask.data.long().sum()
        mask = mask.unsqueeze(2).expand_as(loc_preds)
        masked_loc_preds = loc_preds[mask].view(-1, 4)
        masked_loc_targets = loc_targets[mask].view(-1, 4)
        loc_loss = F.smooth_l1_loss(masked_loc_preds, masked_loc_targets, size_average=False)
        loss = loc_loss / n_obj
        return loss
