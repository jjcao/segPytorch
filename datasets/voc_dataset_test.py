#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 08:10:32 2017

@author: jjcao
"""
import matplotlib.pyplot as plt
import torch
from transforms import Compose, Rescale, RandomCrop, Normalize, ToTensor
from transforms import UnNormalize, FromTensor
from visualizer import show_im_label, show_im_label_batch
from voc_dataset import VocDataset
   
def test_basic_io(dataset_dir, im_num = 3, row = 3, col = 4):
    print('\n\ntest_basic_io')
    voc_dataset = VocDataset(dataset_dir=dataset_dir, transform=None, split='train')
    
#    itor = iter(voc_dataset)
#    for i in range(len(voc_dataset)):          
#        im, lbl = next(itor) 
#        print(i, im.shape, lbl.shape)
#        show_im_label(i, im, lbl, row+1, col)    
#        if i == im_num:
#            plt.show()
#            break
            
    for i, (im, lbl) in enumerate(voc_dataset):
        print(i, im.shape, lbl.shape)
        show_im_label(i, im, lbl, row+1, col)    
        if i == im_num:
            plt.show()
            break
        
#    for i in range(len(voc_dataset)):          
#        im, lbl = voc_dataset[i]  
#        print(i, im.shape, lbl.shape)
#        show_im_label(i, im, lbl, row+1, col)    
#        if i == im_num:
#            plt.show()
#            break
        
        
def test_transforms(dataset_dir, im_num = 2, row = 3, col = 4):
    print('\n\n test_transforms')
    
    composed = Compose([Rescale(256),
                                   RandomCrop(224), 
                                   Normalize(VocDataset.mean_bgr),
                                   ToTensor() ])
#    composed = None
#    composed = Compose([ RandomCrop(224), ToTensor() ])
    voc_dataset = VocDataset(dataset_dir=dataset_dir, split='train', transform=composed)
    

    
    composed1 = Compose([FromTensor(), UnNormalize(VocDataset.mean_bgr) ])
    #composed1 = None
    
    for i in range(len(voc_dataset)):          
        im, lbl  = voc_dataset[i] 
        #print('before transform: {}, {}'.format(im.shape, lbl.shape))
        if composed:
            #im, lbl = composed(im, lbl)
            print('after transform: {}, {}'.format(im.shape, lbl.shape))
        if composed1:
            im, lbl = composed1(im, lbl)
            print('after composed1: {}, {}'.format(im.shape, lbl.shape))
            show_im_label(i, im, lbl, row+1, col) 
        else:
            show_im_label(i, im.numpy().transpose((1, 2, 0)), lbl.numpy(), row+1, col)   
        if i == im_num:
            plt.show()
            break
                    
def test_dataloader(dataset_dir):
    print('\n\ntest_dataloader')
    composed = Compose([Rescale(256),
                                   RandomCrop(224), 
                                   Normalize(VocDataset.mean_bgr),
                                   ToTensor() ])

    dataset = VocDataset(dataset_dir=dataset_dir, split='train', transform=composed)    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, 
                                             shuffle=False, num_workers=4)
  
    

        
    for i_batch, (im_batch, lbl_batch) in enumerate(dataloader):
#    for i_batch, sample_batched in enumerate(dataloader): 
#        im_batch = sample_batched['image']
#        lbl_batch = sample_batched['label'].size()    
        print(i_batch, im_batch.size(), lbl_batch.size())

        #observe 4th batch and stop.
        show_im_label_batch(i_batch, im_batch, lbl_batch) 
        if i_batch == 2:        
            break

      
###############################
####      test 
###############################  
if __name__ == '__main__':    
    dataset_dir = '/Users/jjcao/Documents/jjcao_data/VOCdevkit/VOC2012/'
    test_basic_io(dataset_dir, 3, 2, 4)
    test_transforms(dataset_dir, 3, 2, 4)
    test_dataloader(dataset_dir)