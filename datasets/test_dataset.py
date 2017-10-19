#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2017.10.19

@author: jjcao
"""

from __future__ import print_function, division
import os.path as osp
import os
import numpy as np
import PIL.Image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


###############################
###############################
class TestDataset(Dataset):
    mean_bgr=np.array([104.00698793, 116.66876762, 122.67891434])    
    def __init__(self, root, transform, nclass):
        """
        Args:
            root (string): root path to input images, 
                           such as /Users/jjcao/Documents/input       
            transform (callable): transform to be applied on a sample.
        """
        self.root = root     
        self.nclass = nclass
        self.split = 'test'       
        self.transform = transform        
        self.im_files = os.listdir(root)
#        for i in range(len(self.im_files)):   
#            self.im_files[i] = osp.join(root, self.im_files[i])
                
    def __len__(self):
        return len(self.im_files)

    def __getitem__(self, index):
        data_file = osp.join(self.root, self.im_files[index])
                
        name = osp.splitext(self.im_files[index])[0]
        
        im = PIL.Image.open(data_file)
        im = np.array(im, dtype=np.uint8)   
        lbl = PIL.Image.new('I', im.shape[0:2])
        lbl = np.array(lbl, dtype=np.int32)
         
        if self.transform:
            return (*(self.transform(im, lbl)), name)
        else:   
            return im, lbl, name
  
    def untransform(self, im, lbl):
        im = im.numpy()
        im = im.transpose(1, 2, 0)
        im += self.mean_bgr
        im = im.astype(np.uint8)
        im = im[:, :, ::-1]
        lbl = lbl.numpy()
        return im, lbl 
    
    def get_labels_color(self):
        if self.nclass == 2:
            return np.asarray([[0,0,0], [255,255,255]]) 
        if self.nclass == 21:
            return np.asarray([[0,0,0], [128,0,0], [0,128,0], [128,128,0], [0,0,128], [128,0,128],
                              [0,128,128], [128,128,128], [64,0,0], [192,0,0], [64,128,0], [192,128,0],
                              [64,0,128], [192,0,128], [64,128,128], [192,128,128], [0, 64,0], [128, 64, 0],
                              [0,192,0], [128,192,0], [0,64,128]])
        else:
            raise ValueError
            
    def encode_segmap(self, mask):
        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
        for i, label in enumerate(self.get_labels_color()):
            label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = i
        label_mask = label_mask.astype(int)
        return label_mask


    def decode_segmap(self, temp, plot=False):
        label_colours = self.get_labels_color()
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.nclass):
            r[temp == l] = label_colours[l, 0]
            g[temp == l] = label_colours[l, 1]
            b[temp == l] = label_colours[l, 2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r
        rgb[:, :, 1] = g
        rgb[:, :, 2] = b
        if plot:
            plt.imshow(rgb)
            plt.show()
        else:
            return rgb            