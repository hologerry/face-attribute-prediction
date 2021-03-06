from __future__ import absolute_import, print_function

import torch

__all__ = ['accuracy', 'accuracy_b']


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def accuracy_b(output, target, pseudo, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        # batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        pseudo = pseudo.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct_batch = correct * pseudo
        nonzero = torch.sum(pseudo)

        res = []
        for k in topk:
            correct_k = correct_batch[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / nonzero.item()))
        return res
