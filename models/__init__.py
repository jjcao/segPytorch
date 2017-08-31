#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 00:19:24 2017
todo office choice between vgg16 and resnet-101, etc 
@author: jjcao
"""

import torch

import torchvision.models as models
from models.fcn import FCN32s, FCN16s, FCN8s

def get_model(name, n_classes, checkpoint, cfg):
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
            if name == 'fcn32s':
                vgg16 = models.vgg16(pretrained=True)
    #            from models.vgg import Vgg16
    #            vgg16 = Vgg16(pretrained=True)
                model.init_vgg16_params(vgg16)  
            elif name == 'fcn16s':
                fcn32s = FCN32s(n_classes=n_classes)
                fcn32s.load_state_dict(torch.load(cfg['fcn32s_pretrained_model']))
                model.copy_params_from_fcn32s(fcn32s)
                model.init_vgg16_params(vgg16)
            elif name == 'fcn8s':
                fcn16s = FCN32s(n_classes=n_classes)
                fcn16s.load_state_dict(torch.load(cfg['fcn16s_pretrained_model']))
                model.copy_params_from_fcn16s(fcn16s)
            else:
                raise 'Model {} not available'.format(name)
    else:
        raise 'Model {} not available'.format(name)

    return model, start_epoch,  start_iteration

def _get_model_instance(name):
    return {
        'fcn32s': FCN32s,
        'fcn8s': FCN8s,
        'fcn16s': FCN16s,
#        'unet': unet,
#        'segnet': segnet,
#        'pspnet': pspnet,
#        'linknet': linknet,
    }[name]