#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 00:19:24 2017

@author: jjcao
"""


import torchvision.models as models

from models.fcn import fcn32s

def get_model(name, n_classes):
    model = _get_model_instance(name)

    if name in ['fcn32s', 'fcn16s', 'fcn8s']:
        model = model(n_classes=n_classes)
        vgg16 = models.vgg16(pretrained=True)
        model.init_vgg16_params(vgg16)
    else:
        raise 'Model {} not available'.format(name)

    return model

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