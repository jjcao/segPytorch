#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

hed: Holistically-Nested Edge Detection, iccv 2015 oral

Created on Tue Sep  5 08:48:34 2017

@author: jjcao
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

cfg = {
    'vgg16': [[64, 64],         #conv_block1
              ['M', 128, 128],
              ['M', 256, 256, 256],  #conv_block3
              ['M', 512, 512, 512], 
              ['M', 512, 512, 512]] #conv_block5
    #'resnet18': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
}
                         
def make_layers_vgg16(cfg, n_classes, batch_norm=False):
    in_channels = 3
    kernel_size = 3
    conv_blocks = []
    side_blocks = []
    for i in range(len(cfg)): 
        block = cfg[i]
        layers = []
        for och in block: # och for output_channels
            if och == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, och, kernel_size, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(och), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]

                in_channels = och   
                
        #.in_channels
        side_blocks.append( nn.Conv2d(och, n_classes, 1))    
        conv_blocks.append( nn.Sequential(*layers))    
                
    return conv_blocks, side_blocks

# Hed
class Hed(nn.Module):

    def __init__(self, n_classes=21):
        super(Hed, self).__init__()
        self.n_classes = n_classes
        
        self.conv_blocks, self.side_blocks = make_layers_vgg16(cfg['vgg16'], self.n_classes)  
        self.fuse = nn.Conv2d(5*n_classes, n_classes, 1, 1)

    def forward(self, x):
        input = x
        sidescore = []
        for convblock, sideblock in zip(self.conv_blocks, self.side_blocks):
            conv = convblock(input)
            score = sideblock(conv)
            score = F.upsample_bilinear(score, x.size()[2:])
            sidescore.append(score)  
        
        concat = torch.cat(*sidescore, dim=1)
        fuse = self.fuse(concat)
        return sidescore, fuse

    
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



 