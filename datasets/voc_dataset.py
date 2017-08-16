#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 10:12:31 2017

@author: jjcao
"""

from __future__ import print_function, division
import os.path as osp
import collections

from skimage import io
import numpy as np

from torch.utils.data import Dataset
import torch


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

#plt.ion()   # interactive mode


###############################
###############################
class VocDataset(Dataset):
    """pascal VOC2012 dataset."""

    def __init__(self, dataset_dir='/Users/jjcao/Documents/jjcao_data/VOCdevkit/VOC2012/', 
                 split='train', transform=None):
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
        """todo: could I read it from somewhere"""
        self.n_classes = 21 
        self.transform = transform
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
            
        sample = {'image': im, 'label': lbl}

        if self.transform:
            sample = self.transform(sample)
            
        return sample                
 
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
        image, landmarks = sample['image'], sample['landmarks']

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

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'landmarks': landmarks}
 
    
    
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
        #label = label.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'label': torch.from_numpy(label)}
        


        
    






  



