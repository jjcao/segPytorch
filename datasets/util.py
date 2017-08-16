#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 23:46:38 2017

@author: jjcao
"""
import matplotlib.pyplot as plt
import torch
import numpy as np
from torchvision import utils
from skimage import transform
###############################
# assistant class
###############################     
class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        if new_h >= h: 
            top = 0
        else:
            top = np.random.randint(0, h - new_h)
        if new_w >= w: 
            left = 0
        else:
            left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]
        
        label = label[top: top + new_h,
                      left: left + new_w]
        
        hpad = 0
        wpad = 0
        if new_h > h:
            hpad = new_h - h
        if new_w > w:
            wpad = new_w - w
        if (new_h > h) or (new_w > w):
            image = np.lib.pad(image, ((0,hpad),(0,wpad), (0,0)), 'edge')
            label = np.lib.pad(label, ((0,hpad),(0,wpad)), 'edge')

        return {'image': image, 'label': label}
 
    
class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or tuple): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))
        label = transform.resize(label, (new_h, new_w))

        return {'image': img, 'label': label}
 
class Normalize(object):
    def __init__(self, mean_bgr=np.array([104.00698793, 116.66876762, 122.67891434])):
        self.mean_bgr = mean_bgr    

    def __call__(self, sample):
        img = sample['image']     
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean_bgr
        return {'image': img, 'label': sample['label']}
        #return img, lbl    
    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # image.shape (281, 500, 3) => (3, 281, 500)
        image = image.transpose((2, 0, 1))
        # label.shape (281, 500)
#        return {'image': torch.from_numpy(image).float(),
#                'label': torch.from_numpy(label).long()}
        return {'image': torch.from_numpy(image),
                'label': torch.from_numpy(label)}
        
def show_sample_batch(i_batch, sample_batched):
    """Show image with landmarks for a batch of samples."""
    im_batch, lbl_batch = sample_batched['image'], sample_batched['label']
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
  
#    lbls = torch.ones( (len(lbl_batch), 3, *lbl_batch[0].shape) )   
#    for i in range(len(lbl_batch)):
#        lbls[i] = torch.stack( (lbl_batch[i],lbl_batch[i],lbl_batch[i]), 0);
#    grid = utils.make_grid(lbls)
#    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    plt.show()