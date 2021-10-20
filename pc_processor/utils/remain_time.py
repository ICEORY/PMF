from .avgmeter import RunningAvgMeter

class RemainTime(object):
    def __init__(self, n_epochs):
        self.n_epochs = n_epochs
        self.timer_avg = {}
        self.total_iter = {}
    
    def update(self, cost_time, batch_size=1, mode="Train"):
        if mode not in self.timer_avg.keys():
            self.timer_avg[mode] = RunningAvgMeter()
            self.total_iter[mode] = 0
        self.timer_avg[mode].update(cost_time)
    
    def reset(self):
        self.timer_avg = {}

    def getRemainTime(self, epoch, iters, total_iter, mode="Train"):
        if self.total_iter[mode] == 0:
            self.total_iter[mode] = total_iter

        remain_time = 0
        mode_idx = list(self.timer_avg.keys()).index(mode)
        count = 0
        for k, v in self.timer_avg.items():
            if k == mode:
                remain_iter = (self.n_epochs - epoch) * self.total_iter[k] - iters
            else:
                if count < mode_idx:
                    remain_iter = (self.n_epochs - epoch - 1) * self.total_iter[k]
                else:
                    remain_iter = (self.n_epochs - epoch) * self.total_iter[k]
            count += 1
            remain_time += v.avg * remain_iter
        return remain_time