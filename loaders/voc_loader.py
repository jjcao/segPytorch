#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 10:12:31 2017

@author: jjcao
"""

from __future__ import print_function, division
import os.path as osp
import collections

import torch
import pandas as pd
from skimage import io

import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode


###############################
###############################
class VocDataset(Dataset):
    """pascal VOC2012 dataset."""

    def __init__(self, dataset_dir='/Users/jjcao/Documents/jjcao_data/VOCdevkit/VOC2012/', 
                 split='train'):
        """
        Args:
            dataset_dir (string): Path to pascal VOC2012
            split (string): image name list in VOCdevkit/VOC2012/ImageSets/Segmentation, 
                            such as  'train', 'val' or 'trainval'         
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dataset_dir = dataset_dir
        self.split = split
        self.files = collections.defaultdict(list)
        
        for split in ['train', 'val', 'trainval']:
            imset_file = osp.join(
                dataset_dir, 'ImageSets/Segmentation/%s.txt' % split)
            for did in open(imset_file):
                did = did.strip()
                # input image
                im_file = osp.join(dataset_dir, 'JPEGImages/%s.jpg' % did)
                # label file, i.e. groundth file
                lbl_file = osp.join(
                    dataset_dir, 'SegmentationClassAug/%s.png' % did)
                self.files[split].append({
                    'im': im_file,
                    'lbl': lbl_file,
                })

    def __len__(self):
        return len(self.files[self.split])


    def __getitem__(self, index):
        data_file = self.files[self.split][index]
        im = io.imread(data_file['im'])
        lbl = io.imread(data_file['lbl'])

        lbl[lbl == 255] = -1

        return im, lbl
        
###############################
###############################
def show_im_label(im, lbl, idx):
    """Show image and its label, i.e. groundtruth"""
    ax = plt.subplot(4, 4, 2*idx+1)
    plt.tight_layout()
    #ax.set_title('Sample #{}'.format(idx))
    ax.axis('off')      
    plt.imshow(im)
    plt.pause(0.001)  # pause a bit so that plots are updated
    
    ax = plt.subplot(4, 4, 2*idx+2)
    plt.tight_layout()
    #ax.set_title('Sample #{}'.format(idx))
    ax.axis('off') 
    plt.imshow(lbl)
    plt.pause(0.001)
 
###############################
####      test 
###############################  
if __name__ == '__main__':
    dataset_dir = '/Users/jjcao/Documents/jjcao_data/VOCdevkit/VOC2012/'
    
    voc_dataset = VocDataset(dataset_dir=dataset_dir, split='train')
    fig = plt.figure()
    j = 0
    for i in range(len(voc_dataset)):
#        if i == 0: 
#            continue
          
        im, lbl = voc_dataset[i]   
        print(i, im.shape, lbl.shape)
        show_im_label(im, lbl, j)    
        if j == 7:
            plt.show()
            break
        
        j += 1 
###############################
###############################





  



