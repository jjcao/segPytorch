#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adapted from https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/models/vgg.py
Created on Thu Aug 17 00:19:24 2017
@author: jjcao
"""
import os
#import os.path as osp

from utils import data

import torch
import torchvision

def Vgg16(pretrained=False):
    model = torchvision.models.vgg16(pretrained=False)
    if not pretrained:
        return model
    model_file = _get_vgg16_pretrained_model()
    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict)
    return model


def _get_vgg16_pretrained_model():

    torch_home = os.path.expanduser(os.getenv('TORCH_HOME', '~/.torch'))
    model_dir = os.getenv('TORCH_MODEL_ZOO', os.path.join(torch_home, 'models'))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    filename = 'vgg16_from_caffe.pth'
    cached_file = os.path.join(model_dir, filename)
    
    return data.cached_download(
        url='http://drive.google.com/uc?id=0B9P1L--7Wd2vLTJZMXpIRkVVRFk',
        path = cached_file,
        #path=osp.expanduser('~/data/models/pytorch/vgg16_from_caffe.pth'),
        md5='aa75b158f4181e7f6230029eb96c1b13',
    )
