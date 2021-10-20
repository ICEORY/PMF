import torch 
import torch.nn as nn 
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.epsilon = 1e-6
    
    def forward(self, x, target, mask=None):
        """ 
        compute dice loss
        x: pred, NC or NCHW
        target: N or NHW
        mask: N or NHW
        """
        # mask unavailable points
        if mask is not None:
            target = target * mask
            if mask.dim() == x.dim() -1:
                mask = mask.unsqueeze(1)
            x = x * mask

        n_classes = x.size(1)
        n_batch = x.size(0)
        target = F.one_hot(target, num_classes=n_classes) # shape: NC or NHWC
        # print(target.size())
        # reshape to NC
        if target.dim() > 2:
            target = target.view(-1, n_classes)

        # reshape to NC
        if x.dim() > 2:
            pred = x.view(x.size(0), x.size(1), -1)
            pred = pred.transpose(1, 2)
            pred = pred.contiguous().view(-1, x.size(1))
        else:
            pred = x
        # print(pred.size(), target.size())
        intersect = (pred * target).sum(0) * 2 + self.epsilon
        denominator = (pred + target).sum(0) + self.epsilon
        dice = intersect / denominator
        dice = dice.mean()
        return dice

class ExpLogDiceLoss(nn.Module):
    def __init__(self, gamma=0.3):
        super(ExpLogDiceLoss, self).__init__()
        self.gamma = 0.3
        self.dc = DiceLoss()

    def forward(self, x, target, mask=None):
        dc_loss = self.dc(x, target, mask)
        explog_loss = torch.pow(-dc_loss.clamp(min=1e-6).log(), self.gamma)
        return explog_loss

# if __name__ == "__main__":
#     test_x = torch.rand((3, 4, 4, 4))
#     test_y = torch.randint(0, 4, (3, 4, 4))   
#     criterion = ExpLogDiceLoss()
#     loss = criterion(test_x, test_y)
#     print(loss) 





        
