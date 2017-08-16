#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 08:10:32 2017

@author: jjcao
"""

import matplotlib.pyplot as plt
from torchvision import transforms, utils
import torch
from voc_dataset import *

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
    

  
def test_basic_io(dataset_dir, im_num = 3, row = 3, col = 4):
    voc_dataset = VocDataset(dataset_dir=dataset_dir, split='train')
    for i in range(len(voc_dataset)):          
        sample = voc_dataset[i]  
        im = sample['image']
        label = sample['label']
        print(i, im.shape, label.shape)
        show_im_label(i, im, label, row+1, col)    
        if i == im_num:
            plt.show()
            break

        
def test_transforms(dataset_dir, im_num = 2, row = 3, col = 4):
    voc_dataset = VocDataset(dataset_dir=dataset_dir, split='train')
    #composed = None
    composed = transforms.Compose([ RandomCrop(224), 
                                    ToTensor() ])
    
    for i in range(len(voc_dataset)):          
        sample = voc_dataset[i] 
        print('before transform: ')
        print(sample['image'].shape, sample['label'].shape)
        transformed_sample = composed(sample)
        im = transformed_sample['image']
        label = transformed_sample['label']
        print('after transform: ')
        print(im.shape, label.shape)
        show_im_label(i, im.numpy().transpose((1, 2, 0)), label.numpy(), row+1, col)   
        if i == im_num:
            plt.show()
            break



###############################
############################### 
def show_sample_batch(i_batch, sample_batched):
    """Show image with landmarks for a batch of samples."""
    im_batch, lbl_batch = sample_batched['image'], sample_batched['label']
    #print(lbl_batch.shape)
    
    plt.figure(1)
    grid = utils.make_grid(im_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.title('Batch #{} from dataloader'.format(i_batch))
    
    plt.figure(2)
    lbls = lbl_batch[0].numpy()
    for i in range(1, len(lbl_batch)):
        lbls = np.concatenate( (lbls,lbl_batch[i].numpy()), 1)
    plt.imshow(lbls)
    plt.title('Batch #{} from dataloader'.format(i_batch))
  
#    lbls = torch.ones( (len(lbl_batch), 3, *lbl_batch[0].shape) )   
#    for i in range(len(lbl_batch)):
#        lbls[i] = torch.stack( (lbl_batch[i],lbl_batch[i],lbl_batch[i]), 0);
#    grid = utils.make_grid(lbls)
#    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    plt.show()
            
def test_dataloader(dataset_dir):
    composed = transforms.Compose([RandomCrop(400), 
                                    ToTensor() ])
#    data_transform = transforms.Compose([
#      #transforms.RandomSizedCrop(224),
#      #transforms.RandomHorizontalFlip(),
#      ToTensor(),
##      transforms.ToTensor(),
#      #transforms.Normalize(mean=[0.485, 0.456, 0.406],
#       #                std=[0.229, 0.224, 0.225])
#    ])
    dataset = VocDataset(dataset_dir=dataset_dir, split='train', transform=composed)    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, 
                                             shuffle=True, num_workers=4)
  
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['image'].size(),
                       sample_batched['label'].size())

        #observe 4th batch and stop.
        show_sample_batch(i_batch, sample_batched) 
        if i_batch == 2:        
            break

      
###############################
####      test 
###############################  
if __name__ == '__main__':
    import voc_dataset_test
    dataset_dir = '/Users/jjcao/Documents/jjcao_data/VOCdevkit/VOC2012/'
    #test_basic_io(dataset_dir, 3, 2, 4)
    #test_transforms(dataset_dir, 3, 2, 4)
    test_dataloader(dataset_dir)