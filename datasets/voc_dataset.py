#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 10:12:31 2017
steal lots from https://github.com/wkentaro/pytorch-fcn/ and https://github.com/meetshah1995/pytorch-semseg

@author: jjcao
"""

from __future__ import print_function, division
import os.path as osp
import collections
import numpy as np
from scipy import io
import PIL.Image
from torch.utils.data import Dataset

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


###############################
###############################
class VocDatasetBase(Dataset):
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
    

    def __init__(self, root, transform, split='train'):
        """
        Args:
            root (string): root path to pascal VOC2012, 
                           such as /Users/jjcao/Documents/jjcao_data, 
                           if the actual path for the data is: /Users/jjcao/Documents/jjcao_data/VOC/VOCdevkit/VOC2012/
            split (string): image name list in VOCdevkit/VOC2012/ImageSets/Segmentation, 
                            such as  'train', 'val'         
            transform (callable): transform to be applied on a sample.
        """
        self.root = root
        self.dataset_dir = osp.join(self.root, 'VOC/VOCdevkit/VOC2012')
        
        self.split = split       
        self.transform = transform
        self.files = collections.defaultdict(list)
        
        #for split in ['train', 'val', 'trainval']:
        imset_file = osp.join(
            self.dataset_dir, 'ImageSets/Segmentation/%s.txt' % split)
        self.parse_imset(imset_file)
    
    def parse_imset(self, imset_file):
        for did in open(imset_file):
            did = did.strip()
            # input image
            im_file = osp.join(self.dataset_dir, 'JPEGImages/%s.jpg' % did)
            # label file, i.e. groundth file
            lbl_file = osp.join(
                self.dataset_dir, 'SegmentationClassAug/%s.png' % did)
            self.files[self.split].append({
                'im': im_file,
                'lbl': lbl_file,
                'name': did,
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
    
    def get_labels_color(self):
        return np.asarray([[0,0,0], [128,0,0], [0,128,0], [128,128,0], [0,0,128], [128,0,128],
                              [0,128,128], [128,128,128], [64,0,0], [192,0,0], [64,128,0], [192,128,0],
                              [64,0,128], [192,0,128], [64,128,128], [192,128,128], [0, 64,0], [128, 64, 0],
                              [0,192,0], [128,192,0], [0,64,128]])

class VOC2011ClassSeg(VocDatasetBase):
    def __init__(self, root, transform, split='seg11valid'):
        self.root = root
        self.dataset_dir = osp.join(self.root, 'VOC/VOCdevkit/VOC2012')
        
        self.split = split       
        self.transform = transform
        self.files = collections.defaultdict(list)
        
        pkg_root = osp.join(osp.dirname(osp.realpath(__file__)), '..')
        imset_file = osp.join(
            pkg_root, 'ext/fcn.berkeleyvision.org', 'data/pascal/seg11valid.txt')
        self.parse_imset(imset_file)

class VOC2012ClassSeg(VocDatasetBase):
    """pascal VOC2012 dataset."""
    url = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'  # NOQA
    def __init__(self, root, split='train', transform=False):
        super(VOC2012ClassSeg, self).__init__(
            root, split=split, transform=transform)


class SBDClassSeg(VocDatasetBase):
    # XXX: It must be renamed to benchmark.tar to be extracted.
    url = 'http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz'  # NOQA    
    
    def __init__(self, root, split='train', transform=False):
        self.root = root
        self.dataset_dir = osp.join(self.root, 'VOC/benchmark_RELEASE/dataset')
        self.split = split
        self.transform = transform
        
        self.files = collections.defaultdict(list)
        #for split in ['train', 'val']:
        imset_file = osp.join(self.dataset_dir, '%s.txt' % split)
        for did in open(imset_file):
            did = did.strip()
            im_file = osp.join(self.dataset_dir, 'img/%s.jpg' % did)
            lbl_file = osp.join(self.dataset_dir, 'cls/%s.mat' % did)
            self.files[split].append({
                'im': im_file,
                'lbl': lbl_file,
                'name': did,
            })

    def __getitem__(self, index):
        data_file = self.files[self.split][index]
        # load image
        im_file = data_file['im']
        im = PIL.Image.open(im_file)
        im = np.array(im, dtype=np.uint8)
        # load label
        lbl_file = data_file['lbl']
        mat = io.loadmat(lbl_file)
        lbl = mat['GTcls'][0]['Segmentation'][0].astype(np.int32)
        lbl[lbl == 255] = -1
        
        if self.transform:
            return (*(self.transform(im, lbl)), data_file['name'])
        else:
            return im, lbl, data_file['name']
