#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
based on pytorch-semseg and pytorch-fcn

@ todo: test staged training
@ todo: We experiment with both staged training and all-at-once training.
Created on Thu Aug 17 00:22:19 2017

@author: jjcao
"""

import torch.nn as nn
import torch.nn.functional as F

cfg = {
    '32s-vgg16': [[64, 64],         #conv_block1
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
    conv_blocks = []
    #classifier = []
    for i in range(len(cfg)): 
        block = cfg[i]
        if i == len(cfg)-1: # for classifier block
            # fc6 + fc7 + fc8
            classifier= nn.Sequential(
                        nn.Conv2d(512, 4096, 7),
                        nn.ReLU(inplace=True),
                        nn.Dropout2d(),
                        nn.Conv2d(4096, 4096, 1),
                        nn.ReLU(inplace=True),
                        nn.Dropout2d(),
                        nn.Conv2d(4096, n_classes, 1),) # score 
        else: # for conv block 1, ..., 5 
            layers = []
            for och in block: # och for output_channels
                layers += [nn.Conv2d(in_channels, och, kernel_size, padding = padding)]
                layers += [nn.ReLU(inplace=True)]
                in_channels = och
                padding = 1
                
            layers += [nn.MaxPool2d(2, stride=2, ceil_mode=True)]  
            conv_blocks.append( nn.Sequential(*layers))    
                
    return conv_blocks, classifier

# FCN
class FCN(nn.Module):

    def __init__(self, n_classes=21, learned_billinear=False):
        super(FCN, self).__init__()
        self.learned_billinear = learned_billinear
        self.n_classes = n_classes
        
        self.conv_blocks, self.classifier = make_layers_vgg16(cfg['32s-vgg16'], self.n_classes)  

        # TODO: Add support for learned upsampling
        if self.learned_billinear:
            raise NotImplementedError
            # upscore = nn.ConvTranspose2d(self.n_classes, self.n_classes, 64, stride=32, bias=False)
            # upscore.scale_factor = None

    def forward(self, x):
        input = x
        for conv in self.conv_blocks:
            out = conv(input)
            input = out
            
        score = self.classifier(out)
        return score
    
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
            
# FCN32s
class FCN32s(FCN):
    def __init__(self, n_classes=21, learned_billinear=False):
        super(FCN32s, self).__init__(n_classes, learned_billinear)
        
    def forward(self, x):
        score = super(FCN32s, self).forward(x)
        #import pdb; pdb.set_trace()
        out = F.upsample_bilinear(score, x.size()[2:])
        return out
  
            
# FCN16s
class FCN16s(FCN):
    def __init__(self, n_classes=21, learned_billinear=False):
        super(FCN16s, self).__init__(n_classes, learned_billinear)
        self.score_pool4 = nn.Conv2d(512, self.n_classes, 1)
        
    def forward(self, x):
        score = super(FCN16s, self).forward(x)
        conv4 = self.conv_blocks[3]
        score_pool4 = self.score_pool4(conv4)
        score = F.upsample_bilinear(score, score_pool4.size()[2:])
        score += score_pool4

        out = F.upsample_bilinear(score, x.size()[2:])

        return out
    
    def init_fcn32s_params(self, fcn32s):
        for name, l1 in fcn32s.named_children():
            try:
                l2 = getattr(self, name)
                l2.weight  # skip ReLU / Dropout
            except Exception:
                continue
            assert l1.weight.size() == l2.weight.size()
            assert l1.bias.size() == l2.bias.size()
            l2.weight.data.copy_(l1.weight.data)
            l2.bias.data.copy_(l1.bias.data)
#        for param in model.parameters():
#            print(type(param.data), param.size())

# FCN8s
class FCN8s(FCN):
    def __init__(self, n_classes=21, learned_billinear=False):
        super(FCN8s, self).__init__(n_classes, learned_billinear)
        self.score_pool4 = nn.Conv2d(512, self.n_classes, 1)
        self.score_pool3 = nn.Conv2d(256, self.n_classes, 1)
        
    def forward(self, x):
        score = super(FCN8s, self).forward(x)
        
        score = self.classifier(self.conv_blocks[4])
        score_pool4 = self.score_pool4(self.conv_blocks[3])
        score_pool3 = self.score_pool3(self.conv_blocks[2])

        score = F.upsample_bilinear(score, score_pool4.size()[2:])
        score += score_pool4
        score = F.upsample_bilinear(score, score_pool3.size()[2:])
        score += score_pool3
        out = F.upsample_bilinear(score, x.size()[2:])

        return out
    
    def copy_params_from_fcn16s(self, fcn16s):
        for name, l1 in fcn16s.named_children():
            try:
                l2 = getattr(self, name)
                l2.weight  # skip ReLU / Dropout
            except Exception:
                continue
            assert l1.weight.size() == l2.weight.size()
            l2.weight.data.copy_(l1.weight.data)
            if l1.bias is not None:
                assert l1.bias.size() == l2.bias.size()
                l2.bias.data.copy_(l1.bias.data)