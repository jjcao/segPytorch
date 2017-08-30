#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# todo update rescale, randomeCrop using PIL Image library recommended by pytorch, instead of skimage
# maybe opencv for speed. (skimage is also awesome)

Created on Wed Aug 16 23:46:38 2017

@author: jjcao
"""

import torch
import numpy as np
from skimage import transform
#import PIL.Image   

class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, im, lbl):
        for t in self.transforms:
            im, lbl = t(im, lbl)
        return im, lbl
    
class RandomCrop(object):
    """Crop randomly the image in a sample.
        todo: use PIL image
    Args:
        output_size (tuple or int): Desired output size. If int, square crop is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, im, lbl):
        h, w = im.shape[:2]
        new_h, new_w = self.output_size

        if new_h >= h: 
            top = 0
        else:
            top = np.random.randint(0, h - new_h)
        if new_w >= w: 
            left = 0
        else:
            left = np.random.randint(0, w - new_w)

        im = im[top: top + new_h,
                      left: left + new_w]
        
        lbl = lbl[top: top + new_h,
                      left: left + new_w]
        
        hpad = 0
        wpad = 0
        if new_h > h:
            hpad = new_h - h
        if new_w > w:
            wpad = new_w - w
        if (new_h > h) or (new_w > w):
            im = np.lib.pad(im, ((0,hpad),(0,wpad), (0,0)), 'edge')
            lbl = np.lib.pad(lbl, ((0,hpad),(0,wpad)), 'edge')

        #return {'image': im, 'label': lbl}
        return im, lbl
 
    
class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size       

    def __call__(self, im, lbl):
#        im = PIL.Image.fromarray(im)
#        im = im.resize(self.output_size, PIL.Image.BILINEAR)
#        im = np.array(im)#, dtype=np.uint8)
#        lbl = lbl.resize(self.output_size) #, PIL.Image.BILINEAR)
#        lbl = np.array(lbl)#, dtype=np.int32)
        
        im = transform.resize(im, self.output_size)
        lbl = lbl.astype(np.float64)
        lbl = transform.resize(lbl, self.output_size)

        return im, lbl
 
class Normalize(object):
    def __init__(self, mean_bgr):
        # mean_bgr: np.array([104.00698793, 116.66876762, 122.67891434]
        self.mean_bgr = mean_bgr    

    def __call__(self, im, lbl):    
        im = im[:, :, ::-1]  # RGB -> BGR
        im = im.astype(np.float64)
        im -= self.mean_bgr
        return im, lbl
    
class UnNormalize(object):
    def __init__(self, mean_bgr):
        #mean_bgr=np.array([104.00698793, 116.66876762, 122.67891434])
        self.mean_bgr = mean_bgr    

    def __call__(self, im, lbl):
        im += self.mean_bgr
        im = im.astype(np.uint8)
        im = im[:, :, ::-1]# BGR -> RGB
        return im, lbl  
    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, im, lbl):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # image.shape (281, 500, 3) => (3, 281, 500)
#        print('ToTensor')
#        print(im.shape)
        im = im.transpose(2, 0, 1)
        im = torch.from_numpy(im).float()
        # label.shape (281, 500)
        lbl = torch.from_numpy(lbl).long()
        return im, lbl
        
class FromTensor(object):
    def __call__(self, im, lbl):
        im = im.numpy()
#        print('ToTensor')
#        print(im.shape)
        im = im.transpose(1, 2, 0)
        return im, lbl.numpy()
       
