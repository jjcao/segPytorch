#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 08:10:32 2017

@author: jjcao
"""
import pdb; pdb.set_trace()
import matplotlib.pyplot as plt
#import torch
#from transforms import UnNormalize, FromTensor
from visualizer import show_im_label, show_im_label_batch
from sps_dataset import Sc2ClassSeg
   
def test_basic_io(dataset_dir, im_num = 3, row = 3, col = 4):
    print('\n\ntest_basic_io')
    dataset = Sc2ClassSeg(root=dataset_dir, transform=None, split='sc2_train')           
    for i, (im, lbl) in enumerate(dataset):
        print(i, im.shape, lbl.shape)
        show_im_label(i, im, lbl, row+1, col)    
        if i == im_num:
            plt.show()
            break
        
#    itor = iter(voc_dataset)
#    for i in range(len(voc_dataset)):          
#        im, lbl = next(itor) 
#        print(i, im.shape, lbl.shape)
#        show_im_label(i, im, lbl, row+1, col)    
#        if i == im_num:
#            plt.show()
#            break
        
      
###############################
####      test 
###############################  
if __name__ == '__main__':    
    dataset_dir = '/Users/jjcao/Documents/data'
    test_basic_io(dataset_dir, 3, 2, 4)
    #test_transforms(dataset_dir, 3, 2, 4)
    #test_dataloader(dataset_dir)