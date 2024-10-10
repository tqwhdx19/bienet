# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple, Union
import torch.nn as nn

import numpy as np
import torch
# from mmengine.model import BaseModule
from torch import Tensor

from mmseg.registry import MODELS
from .utils import weighted_loss
from mmdet.models.losses.l2_loss import L2Loss

@weighted_loss
def l2_loss(pred: Tensor, target: Tensor) -> Tensor:
    """L2 loss.

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.

    Returns:
        torch.Tensor: Calculated loss
    """
    assert pred.size() == target.size()
    loss = torch.abs(pred - target)**2
    return loss


@MODELS.register_module()
class L2WeightedLoss(nn.Module):
    """L2 loss.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self,
                 neg_pos_ub: int = -1,
                 pos_margin: float = -1,
                 neg_margin: float = -1,
                 hard_mining: bool = False,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0,
                 loss_name='L2WeightedLoss'):
        super().__init__()
        self.neg_pos_ub = neg_pos_ub
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.hard_mining = hard_mining
        self.reduction = reduction
        self.loss_weight = loss_weight
        self._loss_name = loss_name

    def forward(self,
                pred: Tensor,
                target: Tensor,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[float] = None,
                reduction_override: Optional[str] = None) -> Tensor:
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (float, optional): Average factor that is used to
                average the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')

        reduction = (
            reduction_override if reduction_override else self.reduction)

        weight = weight.clone().float()
        weight = weight/255

        weights = torch.stack([weight, weight, weight], dim=1)
        pred, weight, avg_factor = self.update_weight(pred, target, weights,
                                                      avg_factor)

        # size = weights.size()
        # pos = torch.sum(weights)
        # negtive = size[0] * size[1] - pos
        # weights[weights < 1] = 0.5
        # weights[weights == 1] = 2
        #
        # weighted_loss = weights * (pred - target)
        # loss_bbox = torch.abs(weighted_loss)**2

        loss_bbox = self.loss_weight * l2_loss(
            pred, target, weight, avg_factor=avg_factor)

        return loss_bbox

    def update_weight(self, pred: Tensor, target: Tensor, weight: Tensor,
                      avg_factor: float) -> Tuple[Tensor, Tensor, float]:
        """Update the weight according to targets."""
        if weight is None:
            weight = target.new_ones(target.size())

        invalid_inds = weight <= 0
        target[invalid_inds] = -1
        pos_inds = target == 1
        neg_inds = target == 0

        if self.pos_margin > 0:
            pred[pos_inds] -= self.pos_margin
        if self.neg_margin > 0:
            pred[neg_inds] -= self.neg_margin
        pred = torch.clamp(pred, min=0, max=1)

        num_pos = int((target == 1).sum())
        num_neg = int((target == 0).sum())
        if self.neg_pos_ub > 0 and num_neg / (num_pos +
                                              1e-6) > self.neg_pos_ub:
            num_neg = num_pos * self.neg_pos_ub
            neg_idx = torch.nonzero(target == 0, as_tuple=False)

            if self.hard_mining:
                costs = l2_loss(
                    pred, target, reduction='none')[neg_idx[:, 0],
                                                    neg_idx[:, 1]].detach()
                neg_idx = neg_idx[costs.topk(num_neg)[1], :]
            else:
                neg_idx = self.random_choice(neg_idx, num_neg)

            new_neg_inds = neg_inds.new_zeros(neg_inds.size()).bool()
            new_neg_inds[neg_idx[:, 0], neg_idx[:, 1]] = True

            invalid_neg_inds = torch.logical_xor(neg_inds, new_neg_inds)
            weight[invalid_neg_inds] = 0

        avg_factor = (weight > 0).sum()
        return pred, weight, avg_factor

    @staticmethod
    def random_choice(gallery: Union[list, np.ndarray, Tensor],
                      num: int) -> np.ndarray:
        """Random select some elements from the gallery.

        It seems that Pytorch's implementation is slower than numpy so we use
        numpy to randperm the indices.
        """
        assert len(gallery) >= num
        if isinstance(gallery, list):
            gallery = np.array(gallery)
        cands = np.arange(len(gallery))
        np.random.shuffle(cands)
        rand_inds = cands[:num]
        if not isinstance(gallery, np.ndarray):
            rand_inds = torch.from_numpy(rand_inds).long().to(gallery.device)
        return gallery[rand_inds]

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
