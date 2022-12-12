import torch
import torch.nn.functional as F
from torch import nn

from utils.helper import masked_operation

class FocalLoss(nn.Module):

    def __init__(self, alpha=0.75, gamma=2, reduction='mean', **kwargs):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, label):
        '''
            Usage is same as nn.BCEWithLogits:
            >>> criteria = FocalLoss()
            >>> logits = torch.randn(8, 19, 384, 384)
            >>> lbs = torch.randint(0, 2, (8, 19, 384, 384)).float()
            >>> loss = criteria(logits, lbs)
        '''
        probs = torch.sigmoid(logits)
        coeff = torch.abs(label - probs).pow(self.gamma).neg()
        log_probs = torch.where(logits >= 0, F.softplus(logits, -1, 50), logits - F.softplus(logits, 1, 50))
        log_1_probs = torch.where(logits >= 0, -logits + F.softplus(logits, -1, 50), -F.softplus(logits, 1, 50))
        loss = label * self.alpha * log_probs + (1. - label) * (1. - self.alpha) * log_1_probs
        loss = loss * coeff

        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        if self.reduction == 'batchmean':
            loss = loss.mean(0).sum()
        return loss


def calc_label_loss(inputs, targets, mask=None):
    """
    :param inputs: FloatTensor, in (B, L, 3) or (B, 3)
    :param targets: IntTensor, in (B, L) or (B)
    :param mask: IntTensor, in (B, L) or (B)
    :return:
    """
    if inputs.dim() == 3:
        inputs = inputs.transpose(-2, -1)
    if mask is not None:
        all_loss = F.cross_entropy(inputs, targets, reduction='none')
        return masked_operation(all_loss, mask, dim=1, operation="mean").mean()
    else:
        return F.cross_entropy(inputs, targets)


def calc_kl_loss(origin, augment):
    origin, augment = origin.softmax(-1), augment.softmax(-1)
    return F.kl_div(augment.log(), origin, reduction="batchmean")

def calc_first_loss(outputs, targets):
    """
        :param outputs: logits, features
        :param targets: labels
        :return:
        """
    ce_loss = calc_label_loss(outputs[0], targets)
    total_loss = ce_loss
    return total_loss, {
        "cls": ce_loss.item(),
    }


def calc_grid_loss(grid_logit, grid_target, mask):
    batch_size, max_len = mask.size()
    mask_grid = (mask.unsqueeze(1) * mask.unsqueeze(-1)).reshape(batch_size, -1)
    # (B, 1, L) * (B, L, 1) -> (B, L, L)
    all_loss = F.cross_entropy(grid_logit.permute(0, 3, 1, 2), grid_target,
                               reduction='none', ignore_index=-1).reshape(batch_size, -1)
    # (B, L, L) -> (B, L * L)
    return masked_operation(all_loss, mask_grid, dim=1, operation="mean").mean()


def calc_grid_focal_loss(grid_logit, grid_target, mask):
    from loss.focal_loss import CrossEntropyFocalLoss
    batch_size, max_len = mask.size()
    mask_grid = (mask.unsqueeze(1) * mask.unsqueeze(-1)).reshape(batch_size, -1)
    fl = CrossEntropyFocalLoss(alpha=0.25, gamma=2, weight=torch.tensor([0.1, 0.3, 0.3, 0.3]).to(mask_grid.device), reduction='none', ignore_index=-1)
    all_loss = fl(grid_logit.permute(0, 3, 1, 2), grid_target).reshape(batch_size, -1)
    return masked_operation(all_loss, mask_grid, dim=1, operation="mean").mean()


def calc_major_loss(major_logit, is_major, mask):
    all_loss = F.binary_cross_entropy_with_logits(major_logit, is_major.float(), reduction='none')
    # all_loss = FocalLoss(alpha=0.75, gamma=2, reduction='none')(major_logit, is_major.long())
    return masked_operation(all_loss, mask, dim=1, operation='mean').mean()


def calc_second_loss(outputs, targets, mask, config):
    """
    :param outputs: grid_logits, label_logits, major_conf (B, L)
    :param targets: "grid", "reflection", "label", "is_major" (B, L)
    :return:
    """
    label_loss = calc_label_loss(outputs["label_logit"], targets["label"].long(), mask)
    grid_loss = calc_grid_loss(outputs["grid_logit"], targets["grid"].long(), mask)
    # max_major = outputs["major_conf"].masked_fill(~targets["is_major"], -1).max(1)[0]
    # max_other = outputs["major_conf"].masked_fill((targets["label"] != 1) * (~targets["is_major"]), -1).max(1)[0]
    # major_loss = (max_other + config["margin"] - max_major).clamp_min(0.0).mean()
    major_loss = calc_major_loss(outputs["major_logit"], targets["is_major"], mask)
    total_loss = config["label"] * label_loss + config["grid"] * grid_loss + config["major"] * major_loss
    return total_loss, {
        "total": total_loss.item(),
        "label": label_loss.item(),
        "grid": grid_loss.item(),
        "major": major_loss.item()
    }

