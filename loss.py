import torch
from torch import nn
from torch.nn import functional as F

from . import lovasz_losses as L


class LossBinary:
    """
    Loss defined as BCE - log(soft_jaccard)

    Vladimir Iglovikov, Sergey Mushinskiy, Vladimir Osin,
    Satellite Imagery Feature Detection using Deep Convolutional Neural Network: A Kaggle Competition
    arXiv:1706.06169
    """

    def __init__(self, jaccard_weight=0):
        self.nll_loss = nn.BCEWithLogitsLoss()
        self.jaccard_weight = jaccard_weight

    def __call__(self, outputs, targets):
        loss = self.nll_loss(outputs, targets)

        if self.jaccard_weight:
            eps = 1e-15
            jaccard_target = (targets == 1).float()
            jaccard_output = F.sigmoid(outputs)

            intersection = (jaccard_output * jaccard_target).sum()
            union = jaccard_output.sum() + jaccard_target.sum()

            loss -= self.jaccard_weight * torch.log((intersection + eps) / (union - intersection + eps))

        return loss


class LovaszBinary:
    """

    """

    def __init__(self, ignore=None):
        self.nll_loss = nn.BCEWithLogitsLoss()
        self.ignore = ignore

    def __call__(self, outputs, targets):
        loss = L.lovasz_hinge(outputs, targets, per_image=True, ignore=self.ignore)

        return loss


class LovaszBSE:
    """

    """

    def __init__(self, ignore=None):
        self.nll_loss = nn.BCEWithLogitsLoss()
        self.ignore = ignore

    def __call__(self, outputs, targets):

        bse_loss = self.nll_loss(F.sigmoid(outputs), targets)
        lovasz_loss = L.lovasz_hinge(outputs, targets, per_image=False, ignore=self.ignore)
        loss = 0.9 * lovasz_loss + 0.1 * bse_loss

        return loss


class BCE:
    """
    """

    def __init__(self):
        self.nll_loss = nn.BCELoss()

    def __call__(self, outputs, targets):
        loss = self.nll_loss(F.sigmoid(outputs), targets)

        return loss


def dice_loss(input, target):
    input = torch.sigmoid(input)
    smooth = 1.

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()

    return 1 - ((2. * intersection + smooth) /
                (iflat.sum() + tflat.sum() + smooth))


class FocalLossSimple(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def __call__(self, input, target):
        # Inspired by the implementation of binary_cross_entropy_with_logits
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

        # This formula gives us the log sigmoid of 1-p if y is 0 and of p if y is 1
        invprobs = F.logsigmoid(-input * (target * 2 - 1))
        loss = (invprobs * self.gamma).exp() * loss

        return loss.mean()


