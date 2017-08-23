#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 00:19:24 2017
todo office choice between vgg16 and resnet-101, etc 
@author: jjcao
"""


import torchvision.models as models
import torch
from models.fcn import fcn32s

def get_model(name, n_classes, checkpoint):
    model = _get_model_instance(name)

    start_epoch = 0
    start_iteration = 0
    
    if name in ['fcn32s', 'fcn16s', 'fcn8s']:
        model = model(n_classes=n_classes)  
        if checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint['epoch']
            start_iteration = checkpoint['iteration']
        else:
            vgg16 = models.vgg16(pretrained=True)
            model.init_vgg16_params(vgg16)  
    else:
        raise 'Model {} not available'.format(name)

    return model, start_epoch,  start_iteration

def _get_model_instance(name):
    return {
        'fcn32s': fcn32s,
#        'fcn8s': fcn8s,
#        'fcn16s': fcn16s,
#        'unet': unet,
#        'segnet': segnet,
#        'pspnet': pspnet,
#        'linknet': linknet,
    }[name]