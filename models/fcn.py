#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
based on pytorch-semseg and pytorch-fcn



1. Using upsample_bilinear instead of nn.ConvTranspose2d, then performace is a 1% worse then using nn.ConvTranspose2d.

2. using vgg16_from_caffe.pth provided by https://github.com/wkentaro/pytorch-fcn is better than vgg16-397923af.pth by pytorch

@ todo: We experiment with both staged training and all-at-once training.
Created on Thu Aug 17 00:22:19 2017

@author: jjcao
"""

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch

cfg = {
    'vgg16': [[64, 64],         #conv_block1
                  [128, 128],
                  [256, 256, 256],  #conv_block3
                  [512, 512, 512], 
                  [512, 512, 512], 
                  [4096, 4096]],     #classifier block
    #'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
}


def make_layers_vgg16(cfg, n_classes):
    in_channels = 3
    kernel_size = 3
    padding = 100
    conv_blocks = nn.ModuleList()
    #classifier = []
    for i in range(len(cfg)): 
        block = cfg[i]
        if i == len(cfg)-1: # for classifier block
            # fc6 + fc7 + fc8
            classifier= nn.Sequential(
                        nn.Conv2d(in_channels, block[0], 7),
                        nn.ReLU(inplace=True),
                        nn.Dropout2d(),
                        nn.Conv2d(block[0], block[1], 1),
                        nn.ReLU(inplace=True),
                        nn.Dropout2d(),
                        nn.Conv2d(block[1], n_classes, 1),) # score 
        else: # for conv block 1, ..., 5 
            layers = nn.ModuleList()
            for och in block: # och for output_channels
                layers += [nn.Conv2d(in_channels, och, kernel_size, padding = padding)]
                layers += [nn.ReLU(inplace=True)]
                in_channels = och
                padding = 1
                
            layers += [nn.MaxPool2d(2, stride=2, ceil_mode=True)]  
            conv_blocks.append( nn.Sequential(*layers))    
                
    return conv_blocks, classifier

# https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/surgery.py
def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()

# FCN
class FCN(nn.Module):

    def __init__(self, n_classes=21, learned_billinear=False):
        super(FCN, self).__init__()
        self.learned_billinear = learned_billinear
        self.n_classes = n_classes
        
        self.conv_blocks, self.classifier = make_layers_vgg16(cfg['vgg16'], self.n_classes)  

        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)    
        
# FCN32s
class FCN32s(FCN):
    def __init__(self, n_classes=21, learned_billinear=False):
        super(FCN32s, self).__init__(n_classes, learned_billinear)

        if self.learned_billinear:            
            self.upscore = nn.ConvTranspose2d(self.n_classes, self.n_classes, 64, stride=32, bias=False)
        
        self._initialize_weights()

    def forward(self, x):
        input = x
        for conv in self.conv_blocks:
            out = conv(input)
            input = out
            
        score = self.classifier(out)
        if self.learned_billinear:
            out = self.upscore(score)
            out = out[:, :, 19:19 + x.size()[2], 19:19 + x.size()[3]].contiguous()
        else:
            out = F.upsample_bilinear(score, x.size()[2:])
            
        return out 
    
        
    def init_vgg16_params(self, vgg16, copy_fc8=False):    
        #import pdb; pdb.set_trace()
        #print(len(self.conv_blocks))      
        #print( self.conv_block1.children()[0].weight.size() )
        ranges = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 29]]
        features = list(vgg16.features.children())

        #print('before init')
        #features[0].weight.data - self.conv_block1[0].weight.data
        
        for idx, conv_block in enumerate(self.conv_blocks):
            for l1, l2 in zip(features[ranges[idx][0]:ranges[idx][1]], conv_block):
                if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                    #print (idx, l1, l2)                    
                    assert l1.weight.size() == l2.weight.size()
                    assert l1.bias.size() == l2.bias.size()
                    l2.weight.data = l1.weight.data
                    l2.bias.data = l1.bias.data
        
#        print('after init')
#        features[0].weight.data - self.conv_block1[0].weight.data
#        pdb.set_trace()
           
        for i1, i2 in zip([0, 3], [0, 3]):
            l1 = vgg16.classifier[i1]
            l2 = self.classifier[i2]
            # print type(l1), dir(l1),
            l2.weight.data = l1.weight.data.view(l2.weight.size())
            l2.bias.data = l1.bias.data.view(l2.bias.size())
        n_class = self.classifier[6].weight.size()[0]
        if copy_fc8:
            l1 = vgg16.classifier[6]
            l2 = self.classifier[6]
            l2.weight.data = l1.weight.data[:n_class, :].view(l2.weight.size())
            l2.bias.data = l1.bias.data[:n_class]
            
# FCN16s
class FCN16s(FCN):
    def __init__(self, n_classes=21, learned_billinear=False):
        super(FCN16s, self).__init__(n_classes, learned_billinear)
        self.score_pool4 = nn.Conv2d(512, self.n_classes, 1)
        if self.learned_billinear:                    
            self.upscore2 = nn.ConvTranspose2d(self.n_class, self.n_class, 4, stride=2, bias=False)
            self.upscore16 = nn.ConvTranspose2d(self.n_class, self.n_class, 32, stride=16, bias=False)
        
        self._initialize_weights()

    def forward(self, x):
        conv1 = self.conv_blocks[0](x)
        conv2 = self.conv_blocks[1](conv1)
        conv3 = self.conv_blocks[2](conv2)
        conv4 = self.conv_blocks[3](conv3)
        conv5 = self.conv_blocks[4](conv4)

        score = self.classifier(conv5)
        score_pool4 = self.score_pool4(conv4)

        if self.learned_billinear: 
            upscore2 = self.upscore2(score)# 1/16 
            
            score_pool4 = score_pool4[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]# 1/16
            
            upscore2 += score_pool4
            
            upscore16 = self.upscore16(upscore2)
            out = upscore16[:, :, 27:27 + x.size()[2], 27:27 + x.size()[3]].contiguous()
        else:    
            upscore2 = F.upsample_bilinear(score, score_pool4.size()[2:])
            upscore2 += score_pool4
            out = F.upsample_bilinear(upscore2, x.size()[2:])#.contiguous()
        
        return out       
        
    def init_fcn32s_params(self, fcn32s):
        for block1, block2 in zip(fcn32s.conv_blocks, self.conv_blocks):
            for l1, l2 in zip(block1, block2):
                try:
                    l2.weight  # skip ReLU / Dropout
                except Exception:
                    continue  
                
                assert l2.weight.size() == l1.weight.size()
                assert l2.bias.size() == l1.bias.size()
                l2.weight.data.copy_(l1.weight.data)
                l2.bias.data.copy_(l1.bias.data)                                     

        for l1, l2 in zip(fcn32s.classifier, self.classifier):
            try:
                l2.weight  # skip ReLU / Dropout
            except Exception:
                continue  
            
            assert l2.weight.size() == l1.weight.size()
            assert l2.bias.size() == l1.bias.size()
            l2.weight.data.copy_(l1.weight.data)
            l2.bias.data.copy_(l1.bias.data)
            
# FCN8s
class FCN8s(FCN):
    def __init__(self, n_classes=21, learned_billinear=False):
        super(FCN8s, self).__init__(n_classes, learned_billinear)
        self.score_pool4 = nn.Conv2d(512, self.n_classes, 1)
        self.score_pool3 = nn.Conv2d(256, self.n_classes, 1)
        
        if self.learned_billinear: 
            self.upscore2 = nn.ConvTranspose2d(self.n_class, self.n_class, 4, stride=2, bias=False)
            self.upscore8 = nn.ConvTranspose2d(self.n_class, self.n_class, 16, stride=8, bias=False)
            self.upscore_pool4 = nn.ConvTranspose2d(self.n_class, self.n_class, 4, stride=2, bias=False)
            
        self._initialize_weights()

    def forward(self, x):
        conv1 = self.conv_blocks[0](x)
        conv2 = self.conv_blocks[1](conv1)
        conv3 = self.conv_blocks[2](conv2)
        conv4 = self.conv_blocks[3](conv3)
        conv5 = self.conv_blocks[4](conv4)

        score = self.classifier(conv5)
        score_pool4 = self.score_pool4(conv4)
        score_pool3 = self.score_pool3(conv3)
        
        if self.learned_billinear: 
            upscore2 = self.upscore2(score)# 1/16 
            
            score_pool4 = score_pool4[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]# 1/16
            
            upscore2 += score_pool4# 1/16
            
            upscore_pool4 =self.upscore_pool4(upscore2)# 1/8
    
            score_pool3 = score_pool3[:, :, 9:9 + upscore_pool4.size()[2], 9:9 + upscore_pool4.size()[3]]# 1/8

            score_pool3 += upscore_pool4  # 1/8
                      
            upscore8 = self.upscore8(score_pool3)
            out = upscore8[:, :, 31:31 + x.size()[2], 31:31 + x.size()[3]].contiguous() 
        else:
            score = F.upsample_bilinear(score, score_pool4.size()[2:])
            score += score_pool4
            score = F.upsample_bilinear(score, score_pool3.size()[2:])
            score += score_pool3
            out = F.upsample_bilinear(score, x.size()[2:])
        
        return out
    
    def init_fcn16s_params(self, fcn16s):
        # todo copy more params from fcn16s? via for m in self.modules(): or 
        for block1, block2 in zip(fcn16s.conv_blocks, self.conv_blocks):
            for l1, l2 in zip(block1, block2):
                try:
                    l2.weight  # skip ReLU / Dropout
                except Exception:
                    continue  
                
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data.copy_(l1.weight.data)
                l2.bias.data.copy_(l1.bias.data)                                     

        for l1, l2 in zip(fcn16s.classifier, self.classifier):
            try:
                l2.weight  # skip ReLU / Dropout
            except Exception:
                continue  
            
            assert l1.weight.size() == l2.weight.size()
            assert l1.bias.size() == l2.bias.size()
            l2.weight.data.copy_(l1.weight.data)
            l2.bias.data.copy_(l1.bias.data)