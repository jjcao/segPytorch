#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:06:58 2017

@author: jjcao
"""

from trainers.trainer import Trainer, cross_entropy2d


class TrainerHed(Trainer):

    def __init__(self, cuda, model, optimizer,
                 train_loader, val_loader, out, max_iter, 
                 l_rate=1.0e-10, l_rate_decay = 1.0, lrDecayEvery=10,
                 size_average=False, interval_validate=None):
       super(TrainerHed, self).__init__(cuda, model, optimizer,
                 train_loader, val_loader, out, max_iter, 
                 l_rate, l_rate_decay, lrDecayEvery,
                 size_average, interval_validate)
      

    def compute_loss(self, score, target):
        loss = []
        for i in range(len(score)):
            tmp = cross_entropy2d(score[i], target, size_average=self.size_average) # todo average or not?
            loss += tmp
        return loss    
            
#        for i in range(len(score)):
#            loss[i], grad[i] = bce_loss(output[i], lb.numpy())
#        torch.autograd.backward(output, grad)
#        print( "batch-idx=%d, loss = %.4f"%(batch_idx, np.array(loss).sum()) )
