#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 15:54:05 2017

@author: jjcao
"""
import datetime
import pytz
import os.path as osp
import os
import tqdm
import shutil

import torch
from torch.autograd import Variable
import torch.nn.functional as F
from datasets import transforms

import math
import numpy as np

from utils import utils
import scipy

def cross_entropy2d(input, target, weight=None, size_average=True):
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()
    # log_p: (n, c, h, w)
    log_p = F.log_softmax(input)
    # log_p: (n*h*w, c)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
        loss /= mask.data.sum()
    return loss

class Trainer(object):

    def __init__(self, cuda, model, optimizer,
                 train_loader, val_loader, out, max_iter, 
                 l_rate=1.0e-10, l_rate_decay = 1.0, lrDecayEvery=10,
                 size_average=False, interval_validate=None):
        # max_iter: max number of iterations: in each interation a batch of data are used.
       
        self.cuda = cuda
        
        self.model = model
        self.optim = optimizer

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.timestamp_start = \
            datetime.datetime.now(pytz.timezone('Asia/Shanghai'))
        self.size_average = size_average

        if interval_validate is None:
            self.interval_validate = len(self.train_loader)
        else:
            self.interval_validate = interval_validate

        self.out = out
        if not osp.exists(self.out):
            os.makedirs(self.out)

        self.log_headers = [
            'epoch',
            'iteration',
            'train/loss',
            'train/acc',
            'train/acc_cls',
            'train/mean_iu',
            'train/fwavacc',
            'valid/loss',
            'valid/acc',
            'valid/acc_cls',
            'valid/mean_iu',
            'valid/fwavacc',
            'elapsed_time',
        ]
        if not osp.exists(osp.join(self.out, 'log.csv')):
            with open(osp.join(self.out, 'log.csv'), 'w') as f:
                f.write(','.join(self.log_headers) + '\n')

        self.epoch = 0 # current epoch
        self.iter = 0 # current iteration
        self.max_iter = max_iter
        self.l_rate = l_rate # base learning rate
        self.l_rate_decay = l_rate_decay
        self.lrDecayEvery = lrDecayEvery
        self.best_mean_iu = 0    
                
    def validate(self):
        #Sets the module in evaluation mode.
        #This has any effect only on modules such as Dropout or BatchNorm.
        self.model.eval()
 
        n_class = len(self.val_loader.dataset.class_names)

        val_loss = 0
        visualizations = []
        label_trues, label_preds = [], []
        for batch_idx, (data, target) in tqdm.tqdm(
                enumerate(self.val_loader), total=len(self.val_loader),
                desc='Valid iteration=%d' % self.iter, ncols=80,
                leave=False):
            
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            score = self.model(data)

            loss = cross_entropy2d(score, target, size_average=self.size_average)
            if np.isnan(float(loss.data[0])):
                raise ValueError('loss is nan while validating')
            val_loss += float(loss.data[0]) / len(data)


            imgs = data.data.cpu()
            lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
            lbl_true = target.data.cpu()
            transform = transforms.Compose(
                    [transforms.FromTensor(), 
                     transforms.UnNormalize(self.val_loader.dataset.mean_bgr)])
            for img, lt, lp in zip(imgs, lbl_true, lbl_pred):
                img, lt = transform(img, lt)
                #img, lt = self.val_loader.dataset.untransform(img, lt)
                label_trues.append(lt)
                label_preds.append(lp)
                if len(visualizations) < 9:
                    viz = utils.visualize_segmentation(
                        lbl_pred=lp, lbl_true=lt, img=img, n_class=n_class)
                    visualizations.append(viz)
                    
        metrics = utils.label_accuracy_score(label_trues, label_preds, n_class)

        out = osp.join(self.out, 'visualization_viz')
        if not osp.exists(out):
            os.makedirs(out)
        out_file = osp.join(out, 'iter%012d.jpg' % self.iter)
        scipy.misc.imsave(out_file, utils.get_tile_image(visualizations))

        val_loss /= len(self.val_loader)

        with open(osp.join(self.out, 'log.csv'), 'a') as f:
            elapsed_time = \
                datetime.datetime.now(pytz.timezone('Asia/Tokyo')) - \
                self.timestamp_start
            log = [self.epoch, self.iter] + [''] * 5 + \
                  [val_loss] + list(metrics) + [elapsed_time]
            log = map(str, log)
            f.write(','.join(log) + '\n')

        mean_iu = metrics[2]
        is_best = mean_iu > self.best_mean_iu
        if is_best:
            self.best_mean_iu = mean_iu
        torch.save({
            'epoch': self.epoch,
            'iteration': self.iter,
            'arch': self.model.__class__.__name__,
            'optim_state_dict': self.optim.state_dict(),
            'model_state_dict': self.model.state_dict(),
            'best_mean_iu': self.best_mean_iu,
        }, osp.join(self.out, 'checkpoint.pth.tar'))
        if is_best:
            shutil.copy(osp.join(self.out, 'checkpoint.pth.tar'),
                        osp.join(self.out, 'model_best.pth.tar'))       

    
    def compute_loss(self, score, target):
        loss = cross_entropy2d(score, target, size_average=self.size_average) # todo average or not?
        loss /= len(target) 
        if np.isnan(float(loss.data[0])):
            raise ValueError('loss is nan while training')
            
        return loss
    
    def train_epoch(self):       
        self.model.train()#Sets the module in train mode.

        n_class = len(self.train_loader.dataset.class_names)

        for batch_idx, (data, target) in tqdm.tqdm(
                enumerate(self.train_loader), total=len(self.train_loader),
                desc='Train epoch=%d' % self.epoch, ncols=80, leave=False):
            #import pdb; pdb.set_trace()
            # for resuming
            iter = batch_idx + self.epoch * len(self.train_loader)
            if self.iter != 0 and (iter - 1) != self.iter:
                continue  
            self.iter = iter

            # for validate
            if self.iter % self.interval_validate == 0:
#                print(self.iter, self.interval_validate)
#                print(data.shape)
                self.validate()
            
            # data preprocessing
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            
            # this is not in standard FCN
            #poly_lr_scheduler(self.optim, self.base_l_rate, iter)
            
            
            # optimize
            self.optim.zero_grad()
            score = self.model(data)
            loss = self.compute_loss(score, target)                
            loss.backward()
            self.optim.step()

            # logging
            metrics = []
            lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
            lbl_true = target.data.cpu().numpy()
            for lt, lp in zip(lbl_true, lbl_pred):
                acc, acc_cls, mean_iu, fwavacc = utils.label_accuracy_score(
                        [lt], [lp], n_class=n_class)
                metrics.append((acc, acc_cls, mean_iu, fwavacc))
            metrics = np.mean(metrics, axis=0)

            with open(osp.join(self.out, 'log.csv'), 'a') as f:
                elapsed_time = (
                    datetime.datetime.now(pytz.timezone('Asia/Shanghai')) -
                    self.timestamp_start).total_seconds()
                log = [self.epoch, self.iter] + [loss.data[0]] + \
                    metrics.tolist() + [''] * 5 + [elapsed_time]
                log = map(str, log)
                f.write(','.join(log) + '\n')

            if self.iter >= self.max_iter:
                break
    

    def train(self):
        #len(self.train_loader) = |images|/batch_size
        max_epoch = int(math.ceil(1. * self.max_iter / len(self.train_loader)))
        for epoch in tqdm.trange(self.epoch, max_epoch, desc='Train', ncols=80):
            self.epoch = epoch
            if epoch % self.lrDecayEvery == 0:
                self.l_rate = self.l_rate * self.l_rate_decay
                for param_group in self.optim.param_groups:
                    param_group['lr'] = self.l_rate

            self.train_epoch()
            if self.iter >= self.max_iter:
                break