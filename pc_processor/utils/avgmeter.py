class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class RunningAvgMeter(object):
    """Computes and stores the running average and current value
    avg = hist_val * alpha + (1-alpha) * curr_val
    """

    def __init__(self, alpha=0.95):
        self.is_init = False
        self.alpha = alpha
        assert (alpha<=1 and alpha>=0), "alpha should be [0, 1]"
        self.reset()

    def reset(self):
        self.is_init = False
        self.avg = 0

    def update(self, val):
        if self.is_init:
            self.avg = self.avg * self.alpha + (1-self.alpha) * val
        else:
            self.avg = val
            self.is_init = True