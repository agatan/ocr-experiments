import torch.nn as nn
import torch.nn.functional as F


class Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def focal_loss(self, x, y):
        '''
        Args:
            x: torch.Tensor [N]
            y: torch.Tensor [N]
        '''
        alpha = 0.25
        gamma = 2
        p = x.sigmoid()
        pt = p * y + (1 - p) * (1 - y)
        w = alpha * y + (1 - alpha) * (1 - y)
        w = w * (1 - pt).pow(gamma)
        return F.binary_cross_entropy_with_logits(x, y, w, size_average=False)

    def forward(self, loc_preds, loc_targets, conf_preds, mask):
        '''
        Args:
            loc_preds: [batch_size, #anchors, 4]
            loc_targets: [batch_size, #anchors, 4]
            mask: [batch_size, #anchors]
        '''
        batch_size, n_boxes, _ = loc_targets.size()
        n_obj = mask.data.long().sum()
        mask_exp = (mask > 0).unsqueeze(2).expand_as(loc_preds)
        masked_loc_preds = loc_preds[mask_exp].view(-1, 4)
        masked_loc_targets = loc_targets[mask_exp].view(-1, 4)
        loc_loss = F.smooth_l1_loss(masked_loc_preds, masked_loc_targets, size_average=False)

        conf_targets = mask.unsqueeze(2).expand_as(conf_preds).float()
        conf_loss = self.focal_loss(conf_preds.view(-1), conf_targets.view(-1))

        loss = (loc_loss + conf_loss) / n_obj
        return loss
