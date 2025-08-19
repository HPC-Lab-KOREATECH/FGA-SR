import math
from functools import lru_cache

import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class FL1Loss(nn.Module):
    """Frequency domain L1 Loss"""
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(FL1Loss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target):
        f_pred = torch.fft.rfft2(pred, norm='ortho')
        f_target = torch.fft.rfft2(target, norm='ortho')

        f_pred_real = torch.view_as_real(f_pred)
        f_tgt_real = torch.view_as_real(f_target)

        loss = F.l1_loss(f_pred_real, f_tgt_real, reduction=self.reduction)
        return self.loss_weight * loss