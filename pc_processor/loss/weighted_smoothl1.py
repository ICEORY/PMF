import torch
import torch.nn as nn

class WeightedSmoothL1Loss(nn.Module):
    def __init__(self, sigma=3):
        super(WeightedSmoothL1Loss, self).__init__()
        self.sigma = sigma

    def forward(self, x, target, weight=None, mask=None):
        diff = (x - target).abs()
        beta = 1 / (self.sigma ** 2)
        cond = diff < beta
        loss = torch.where(cond, 0.5 * diff.pow(2)/beta, diff - 0.5*beta)
        if weight is not None:
            loss = loss * weight
        if mask is not None:
            mask = mask.expand_as(loss)
            loss = loss * mask
            return loss.sum() / mask.sum()
        else:
            return loss.mean()

# if __name__ == "__main__":
#     criterion = WeightedSmoothL1Loss(sigma=10)
#     test_data = torch.rand(10, 10)
#     test_target = torch.rand(10, 10)
#     mask = test_data.ge(0.5)
#     weight = torch.rand(10, 10)
#     loss = criterion(test_data, test_target, weight=weight, mask=mask)
#     print(loss)