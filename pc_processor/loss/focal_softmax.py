import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FocalSoftmaxLoss(nn.Module):
    def __init__(self, n_classes, gamma=1, alpha=0.8, softmax=True):
        super(FocalSoftmaxLoss, self).__init__()
        self.gamma = gamma
        self.n_classes = n_classes

        if isinstance(alpha, list):
            assert len(alpha) == n_classes, "len(alpha)!=n_classes: {} vs. {}".format(
                len(alpha), n_classes)
            self.alpha = torch.Tensor(alpha)
        elif isinstance(alpha, np.ndarray):
            assert alpha.shape[0] == n_classes, "len(alpha)!=n_classes: {} vs. {}".format(
                len(alpha), n_classes)
            self.alpha = torch.from_numpy(alpha)
        else:
            assert alpha < 1 and alpha > 0, "invalid alpha: {}".format(alpha)
            self.alpha = torch.zeros(n_classes)
            self.alpha[0] = alpha
            self.alpha[1:] += (1-alpha)
        self.softmax = softmax

    def forward(self, x, target, mask=None):
        """compute focal loss
        x: N C or NCHW
        target: N, or NHW

        Args:
            x ([type]): [description]
            target ([type]): [description]
        """

        if x.dim() > 2:
            pred = x.view(x.size(0), x.size(1), -1)
            pred = pred.transpose(1, 2)
            pred = pred.contiguous().view(-1, x.size(1))
        else:
            pred = x

        target = target.view(-1, 1)

        if self.softmax:
            pred_softmax = F.softmax(pred, 1)
        else:
            pred_softmax = pred
        pred_softmax = pred_softmax.gather(1, target).view(-1)
        pred_logsoft = pred_softmax.clamp(1e-6).log()
        self.alpha = self.alpha.to(x.device)
        alpha = self.alpha.gather(0, target.squeeze())
        loss = - (1-pred_softmax).pow(self.gamma)
        loss = loss * pred_logsoft * alpha
        if mask is not None:
            if len(mask.size()) > 1:
                mask = mask.view(-1)
            loss = (loss * mask).sum() / mask.sum()
            return loss
        else:
            return loss.mean()


if __name__ == "__main__":
    criterion = FocalSoftmaxLoss(n_classes=10, gamma=1, alpha=0.8)
    target = torch.arange(0, 10)
    print(target)
    test_input = torch.rand(10, 10)
    mask = torch.ones(10)
    mask[4] = 0
    loss = criterion(test_input, target, mask)
    print(loss)
