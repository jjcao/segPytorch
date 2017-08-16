#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 08:10:32 2017

@author: jjcao
"""

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
        
            
def test_dataloader(dataset_dir):
    composed = transforms.Compose([Rescale(256),
                                   RandomCrop(224), 
                                   Normalize(),
                                   ToTensor() ])

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
    import matplotlib.pyplot as plt
    from torchvision import transforms
    import torch
    from voc_dataset import VocDataset
    from util import show_sample_batch, Rescale, RandomCrop, Normalize, ToTensor
    
    dataset_dir = '/Users/jjcao/Documents/jjcao_data/VOCdevkit/VOC2012/'
    #test_basic_io(dataset_dir, 3, 2, 4)
    #test_transforms(dataset_dir, 3, 2, 4)
    test_dataloader(dataset_dir)