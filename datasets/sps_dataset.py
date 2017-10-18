#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 16:51:03 2017

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
class SpsDatasetBase(Dataset):
    #n_classes = len(class_names) = 2
    class_names = np.array([
        'background',
        'foreground',
    ])
    mean_bgr=np.array([104.00698793, 116.66876762, 122.67891434])# todo 亚芳，张媛??
    

    def __init__(self, root, transform, split='sc2_train'):
        """
        Args:
            root (string): root path to shoe print, 
                           such as /Users/jjcao/Documents/jjcao_data, 
                           if the actual path for the data is: /Users/jjcao/Documents/jjcao_data/foot_data
            split (string): image name list in foot_data, 
                            such as 'train_sc2', 'test_sc', 'train_sus', 'test_sus'       
            transform (callable): transform to be applied on a sample.
        """
        self.root = root
        dataset_dir = osp.join(self.root, 'foot_data')
        
        self.split = split       
        self.transform = transform
        self.files = collections.defaultdict(list)
        

        #sc2/origin/1.jpg
        #sc2/origin/10.jpg
        #for split in ['sc2_train', 'sc_val', 'sc_test', 'sus_train', 'sus_val', 'sus_test']:
        imset_file = osp.join(dataset_dir, '%s.txt' % split)
        for did in open(imset_file):
            did = did.strip()
            rpath = osp.split(osp.split(did)[0])[0]
            break
        
        #import pdb; pdb.set_trace()
        for did in open(imset_file):
            did = did.strip()
            # input image
            im_file = osp.join(dataset_dir, did)
            
            # label file, i.e. groundth file 
            name = osp.basename(did)         
            name = osp.splitext(name)[0]
            lbl_file = osp.join(dataset_dir, '%s/end/%s.png' % (rpath, name) )
            self.files[split].append({
                'im': im_file,
                'lbl': lbl_file,
                'name': name,
            })

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        data_file = self.files[self.split][index]
        
        im = PIL.Image.open(data_file['im'])
        im = np.array(im, dtype=np.uint8)
        lbl = PIL.Image.open(data_file['lbl'])
        lbl = np.array(lbl, dtype=np.int32)
        lbl[lbl == 0] = -1
        lbl[lbl == 1] = 0
         
        if self.transform:
            return (*(self.transform(im, lbl)), data_file['name'])
        else:   
             return im, lbl, data_file['name']
        #return {'image': im, 'label': lbl}
  
    def untransform(self, im, lbl):
        im = im.numpy()
        im = im.transpose(1, 2, 0)
        im += self.mean_bgr
        im = im.astype(np.uint8)
        im = im[:, :, ::-1]
        lbl = lbl.numpy()
        return im, lbl
    
class Sc2ClassSeg(SpsDatasetBase):
    def __init__(self, root, split='sc2_train', transform=False):
        super(Sc2ClassSeg, self).__init__(
            root, split=split, transform=transform)    

class ScValClassSeg(SpsDatasetBase):
    def __init__(self, root, split='sc_val', transform=False):
        super(ScValClassSeg, self).__init__(
            root, split=split, transform=transform)  
        
class ScTestClassSeg(SpsDatasetBase):
    def __init__(self, root, split='sc_test', transform=False):
        super(ScTestClassSeg, self).__init__(
            root, split=split, transform=transform)          