#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 08:48:01 2017

@author: jjcao
"""

import matplotlib.pyplot as plt
from torchvision import utils
import numpy as np

def show_im_label(idx, image, label, row = 4, col = 4):
    """Show image and its label, i.e. groundtruth"""
    assert (2*idx+1) < row*col
    
    ax = plt.subplot(row, col, 2*idx+1)
    plt.tight_layout()
    ax.axis('off')      
    plt.imshow(image)
    plt.pause(0.001)  # pause a bit so that plots are updated
    
    ax = plt.subplot(row, col, 2*idx+2)
    plt.tight_layout()
    ax.axis('off') 
    plt.imshow(label)
    plt.pause(0.001)
    
def show_im_label_batch(i_batch, im_batch, lbl_batch):
    """Show a batch of image and labels."""
    #print(lbl_batch.shape)
    
    plt.figure(1)
    mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
#    for i in range(len(im_batch)):
#        im_batch[i] += mean_bgr 
    grid = utils.make_grid(im_batch)
    tmp = grid.numpy().transpose((1, 2, 0)) + mean_bgr
    tmp = tmp[:, :, ::-1]
    plt.imshow(tmp )
    plt.title('Batch #{} from dataloader'.format(i_batch))
    
    plt.figure(2)
    lbls = lbl_batch[0].numpy()
    for i in range(1, len(lbl_batch)):
        lbls = np.concatenate( (lbls,lbl_batch[i].numpy()), 1)
    plt.imshow(lbls)
    plt.title('Batch #{} from dataloader'.format(i_batch))

    plt.show()