import torch
import torch.nn as nn
import torch.nn.functional as F


class BFocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            CE_loss_batch = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            CE_loss_batch = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-CE_loss_batch)
        F_loss = self.alpha * (1 - pt)**self.gamma * CE_loss_batch

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, inputs, targets):
        CE_loss_batch = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-CE_loss_batch)
        F_loss = self.alpha * (1 - pt)**self.gamma * CE_loss_batch

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss
