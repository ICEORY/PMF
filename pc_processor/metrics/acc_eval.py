#!/usr/bin/env python3

# This file is covered by the LICENSE file in the root of this project.

import torch 

class AccEval(object):
    def __init__(self, topk=(1, ), is_distributed=False):
        self.topk = topk 
        self.is_distributed = is_distributed
    
    def getAcc(self, output, target):
        maxk = max(self.topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        if self.is_distributed:
            correct = correct.cuda()
            batch_size = torch.Tensor([batch_size]).cuda()
            torch.distributed.barrier()
            torch.distributed.all_reduce(correct)
            torch.distributed.all_reduce(batch_size)
            correct = correct.to(target)
            batch_size = batch_size.item()
        for k in self.topk:
            correct_k = correct[:k].float().sum()
            acc = correct_k.mul_(100.0/batch_size)
            res.append(acc)
        return res