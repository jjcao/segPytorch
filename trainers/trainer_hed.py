#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:06:58 2017

@author: jjcao
"""

from trainers.trainer import Trainer

import numpy as np

class TrainerHed(Trainer):

    def __init__(self, cuda, model, optimizer,
                 train_loader, val_loader, out, max_iter, 
                 l_rate=1.0e-10, l_rate_decay = 1.0, lrDecayEvery=10,
                 size_average=False, interval_validate=None):
       super(TrainerHed, self).__init__(cuda, model, optimizer,
                 train_loader, val_loader, out, max_iter, 
                 l_rate, l_rate_decay, lrDecayEvery,
                 size_average, interval_validate)
      
    def optimize(self, data, target):
        import pdb; pdb.set_trace()
        self.optim.zero_grad()
        fuse_score, side_scores = self.model(data)
        
        loss = []
        for i in range(len(side_scores)):
            tmp = self.cross_entropy2d(side_scores[i], target, size_average=self.size_average) # todo average or not?
            loss += tmp
        tmp = self.cross_entropy2d(fuse_score, target, size_average=self.size_average)     
        loss += tmp

        loss /= len(target) 
        if np.isnan(float(loss.data[0])):
            raise ValueError('loss is nan while training')
                       
        loss.backward()
        self.optim.step()
     
        return fuse_score, loss

