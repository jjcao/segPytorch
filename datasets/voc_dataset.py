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
from torch.utils.data import Dataset


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

#plt.ion()   # interactive mode


###############################
###############################
class VocDataset(Dataset):
    """pascal VOC2012 dataset."""

    #todo: could I read it from somewhere 
    n_classes = 21


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
 
        


        
    






  



