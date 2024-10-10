import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import VGG19
from mmseg.registry import MODELS

def _expand_onehot_labels_dice(pred: torch.Tensor,
                               target: torch.Tensor) -> torch.Tensor:
    """Expand onehot labels to match the size of prediction.

    Args:
        pred (torch.Tensor): The prediction, has a shape (N, num_class, H, W).
        target (torch.Tensor): The learning label of the prediction,
            has a shape (N, H, W).

    Returns:
        torch.Tensor: The target after one-hot encoding,
            has a shape (N, num_class, H, W).
    """
    num_classes = pred.shape[1]
    one_hot_target = torch.clamp(target, min=0, max=num_classes)
    one_hot_target = torch.nn.functional.one_hot(one_hot_target,
                                                 num_classes + 1)
    one_hot_target = one_hot_target[..., :num_classes].permute(0, 3, 1, 2)
    return one_hot_target


@MODELS.register_module()
class L1():
    def __init__(self, ):
        self.calc = torch.nn.L1Loss()

    def __call__(self, x, y):
        return self.calc(x, y)

@MODELS.register_module()
class Perceptual(nn.Module):
    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super(Perceptual, self).__init__()
        self.vgg = VGG19().cuda()
        self.criterion = torch.nn.L1Loss()
        self.weights = weights

    def __call__(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        content_loss = 0.0
        prefix = [1, 2, 3, 4, 5]
        for i in range(5):
            content_loss += self.weights[i] * self.criterion(
                x_vgg[f'relu{prefix[i]}_1'], y_vgg[f'relu{prefix[i]}_1'])
        return content_loss

@MODELS.register_module()
class Style(nn.Module):
    def __init__(self):
        super(Style, self).__init__()
        self.vgg = VGG19().cuda()
        self.criterion = torch.nn.L1Loss()

    def compute_gram(self, x):
        b, c, h, w = x.size()
        f = x.view(b, c, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * c)
        return G

    def __call__(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        style_loss = 0.0
        prefix = [2, 3, 4, 5]
        posfix = [2, 4, 4, 2]
        for pre, pos in list(zip(prefix, posfix)):
            style_loss += self.criterion(
                self.compute_gram(x_vgg[f'relu{pre}_{pos}']), self.compute_gram(y_vgg[f'relu{pre}_{pos}']))
        return style_loss


@MODELS.register_module()
class L1StylePerceptualLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 activate=True,
                 reduction='mean',
                 naive_dice=False,
                 loss_weight=1.0,
                 ignore_index=255,
                 eps=1e-3,
                 loss_name='l1_style_perceptual_loss'):

        super().__init__()
        self.use_sigmoid = use_sigmoid
        self.reduction = reduction
        self.naive_dice = naive_dice
        self.loss_weight = loss_weight
        self.eps = eps
        self.activate = activate
        self.ignore_index = ignore_index
        self._loss_name = loss_name

        self.loss_l1 = L1()
        self.style = Style()
        self.loss_perceptual = Perceptual()
        self.rec_loss_weight = '1*L1+250*Style+0.1*Perceptual'

    def forward(self,
                pred,
                target,
                weight=None,
                reduction_override=None,
                **kwargs):

        one_hot_target = target
        if (pred.shape != target.shape):
            one_hot_target = _expand_onehot_labels_dice(pred, target)
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        if self.activate:
            if self.use_sigmoid:
                pred = pred.sigmoid()
            elif pred.shape[1] != 1:
                # softmax does not work when there is only 1 class
                pred = pred.softmax(dim=1)

        loss_l1_part = self.loss_l1(pred, one_hot_target)
        loss_style_part = self.style(pred, one_hot_target)
        loss_perceptual_pat = self.loss_perceptual(pred, one_hot_target)

        loss = 1 * loss_l1_part + 250 * loss_style_part + 0.1 * loss_perceptual_pat
        loss_total = loss * self.loss_weight

        return loss_total

    @property
    def loss_name(self):
        return self._loss_name

# rec_loss = '1*L1+250*Style+0.1*Perceptual'
# losses = list(rec_loss.split('+'))
#
# rec_loss = {}
# for l in losses:
#     weight, name = l.split('*')
#     rec_loss[name] = float(weight)
