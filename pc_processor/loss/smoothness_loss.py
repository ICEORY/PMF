import torch
import torch.nn as nn
import torch.nn.functional as F
from .weighted_smoothl1 import WeightedSmoothL1Loss

class SmoothnessLoss(nn.Module):
    def __init__(self, size_average=True):
        super(SmoothnessLoss, self).__init__()
        self.kernel = [
            [-1/8, -1/8, -1/8],
            [-1/8, 1, -1/8],
            [-1/8, -1/8, -1/8]]
        self.kernel_size = 3
        self.size_average = size_average

    def forward(self, x):
        # assert x.size(1) == 1, "only support one channel"
        weight = torch.FloatTensor(self.kernel).to(x.device).unsqueeze(0).unsqueeze(0)
        weight = weight.expand(x.size(1), 1, self.kernel_size,  self.kernel_size)
        div_map = F.conv2d(x, weight, groups=x.size(1), padding=1)
        if self.size_average:
            return div_map.abs().mean()
        else:
            return div_map

class GradGuideLoss(nn.Module):
    def __init__(self, mode="SmoothL1"):
        super(GradGuideLoss, self).__init__()
        if mode == "SmoothL1":
            self.criterion = WeightedSmoothL1Loss()
        else:
            raise NotImplementedError
    
        self.gradient_module = SmoothnessLoss(False)
    
    def forward(self, x, target):
        with torch.no_grad():
            t_grad = self.gradient_module(target)
        x_grad = self.gradient_module(x) 

        loss = self.criterion(x_grad, t_grad)
        return loss   