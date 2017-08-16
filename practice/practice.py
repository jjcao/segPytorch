#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 14:08:00 2017

@author: jjcao
"""

import torch
import numpy as np
from skimage import io
import matplotlib.pyplot as plt


b = np.arange(12).reshape(2,2,3)
print(b.shape)
tmp = np.concatenate( (b[:,:,0], b[:,:,1]), 1)
tmp = np.concatenate( (tmp, b[:,:,2]), 1)
print(tmp)

b = b[:, :, ::-1]
print(b.shape)
tmp = np.concatenate( (b[:,:,0], b[:,:,1]), 1)
tmp = np.concatenate( (tmp, b[:,:,2]), 1)
print(tmp)
#img = img[:, :, ::-1]  # RGB -> BGR

def test_image_cat():
    dataset_dir = '/Users/jjcao/Documents/jjcao_data/VOCdevkit/VOC2012/'
    lbl_file = osp.join(dataset_dir, 'SegmentationClassAug/2007_000033.png')
    lbl = io.imread(lbl_file)
    lbls = lbl
    for i in range(1,3):
        lbls = np.concatenate( (lbls,lbl), 1)
    
    plt.imshow(lbls)
    plt.show()
    #a = np.zeros(2,2, dtype=np.int)

def test_pad():
    a = torch.ones(2,2,3)
    a = a.numpy()
    print(a)
    print(a.shape)
    
    b = np.arange(12).reshape(2,2,3)
    print(b)
    print(b.shape)
    b[:,:,0]
    c = np.lib.pad(b, ((0,1),(0,1), (0,0)), 'edge')
    print(c)
    print(c.shape)
    c[:,:,0]

def test_tensor():
    x = torch.eye(4,5)
    print(x[:5,0:6])
    y = torch.stack((x,x,x),0)
    #y = torch.cat((x,x,x),2)
    print(y.shape)
    print(y)
    
    z = torch.stack((y,y,y,y),0)
    print(z.shape)
    
    z1 = torch.ones( (4,*y.shape) )
    for i in range(0,3):
        z1[i] = y
    print(z1.shape)
        
    z2 = torch.cat((z, z1), 0)
    print(z2.shape)