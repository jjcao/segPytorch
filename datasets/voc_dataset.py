#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 10:12:31 2017

@author: jjcao
"""

from __future__ import print_function, division
import os.path as osp
import collections
import numpy as np

import PIL.Image
from torch.utils.data import Dataset

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


###############################
###############################
class VocDataset(Dataset):
    """pascal VOC2012 dataset."""

    #todo: could I read it from somewhere 
    #n_classes = len(class_names) = 21
    class_names = np.array([
        'background',
        'aeroplane',
        'bicycle',
        'bird',
        'boat',
        'bottle',
        'bus',
        'car',
        'cat',
        'chair',
        'cow',
        'diningtable',
        'dog',
        'horse',
        'motorbike',
        'person',
        'potted plant',
        'sheep',
        'sofa',
        'train',
        'tv/monitor',
    ])
    mean_bgr=np.array([104.00698793, 116.66876762, 122.67891434])
    

    def __init__(self, dataset_dir, transform, split='train'):
        """
        Args:
            dataset_dir (string): Path to pascal VOC2012, 
                                  such as /Users/jjcao/Documents/jjcao_data/VOCdevkit/VOC2012/
            split (string): image name list in VOCdevkit/VOC2012/ImageSets/Segmentation, 
                            such as  'train', 'val' or 'trainval'         
            transform (callable): transform to be applied on a sample.
        """
        self.dataset_dir = dataset_dir
        self.split = split       
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
        
        im = PIL.Image.open(data_file['im'])
        im = np.array(im, dtype=np.uint8)
        lbl = PIL.Image.open(data_file['lbl'])
        lbl = np.array(lbl, dtype=np.int32)
        lbl[lbl == 255] = -1
         
        if self.transform:
            return self.transform(im, lbl)
        else:   
             return im, lbl
        #return {'image': im, 'label': lbl}
  
    def untransform(self, img, lbl):
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        img += self.mean_bgr
        img = img.astype(np.uint8)
        img = img[:, :, ::-1]
        lbl = lbl.numpy()
        return img, lbl
    
#    def __iter__(self):
#        self.n = 0
#        self.max = self.__len__()
#        
#        return self
#    
#    def __next__(self):
#        if self.n <= self.max:
#            im, lbl = self.__getitem__(self.n)
#            self.n += 1
#        else:
#            raise StopIteration
#            
#        return im, lbl


        
    






  



