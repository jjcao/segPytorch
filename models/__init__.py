#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
from pytorch-semseg


Created on Thu Aug 17 00:19:24 2017
todo office choice between vgg16 and resnet-101, etc 
@author: jjcao
"""

import torch

from models.fcn import FCN32s, FCN16s, FCN8s
from models.segnet import Segnet
from models.unet import Unet
from models.pspnet import Pspnet
from models.linknet import Linknet
from models.hed import Hed

model_dict = {
    'fcn32s': FCN32s,
    'fcn8s': FCN8s,
    'fcn16s': FCN16s,
    'unet': Unet,
    'segnet': Segnet,
    'pspnet': Pspnet,
    'linknet': Linknet,
    'hed': Hed,
}
    
def get_model(name, n_classes, checkpoint, args):
    model = model_dict[name]

    start_epoch = 0
    start_iteration = 0
    
    if name in model_dict.keys():
        model = model(n_classes=n_classes) 
        
        if checkpoint:
#            if torch.cuda.is_available():
#                model = torch.nn.DataParallel(model).cuda()
            
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint['epoch']
            start_iteration = checkpoint['iteration']
        else:  
            #import pdb; pdb.set_trace()
            if name in ['fcn32s', 'segnet']:
                # the vgg16 pretrained model (vgg16-397923af.pth) provided by pytorch is far weak than 
                # vgg16_from_caffe.pth provided by https://github.com/wkentaro/pytorch-fcn
                #vgg16 = models.vgg16(pretrained=True)
                
                from models.vgg import Vgg16               
                vgg16 = Vgg16(pretrained=True)
                model.init_vgg16_params(vgg16)  
            elif name == 'fcn16s':
                fcn32s = FCN32s(n_classes=n_classes)
                
                if torch.cuda.is_available():
                    base = torch.load(args['fcn32s_pretrained_model']) 
                else:
                    base = torch.load(args['fcn32s_pretrained_model'], 
                                      map_location=lambda storage, loc: storage)
        
                fcn32s.load_state_dict(base['model_state_dict'])
                model.init_fcn32s_params(fcn32s)
            elif name == 'fcn8s':
                fcn16s = FCN16s(n_classes=n_classes)
                
                if torch.cuda.is_available():
                    base = torch.load(args['fcn32s_pretrained_model']) 
                else:
                    base = torch.load(args['fcn32s_pretrained_model'], 
                                      map_location=lambda storage, loc: storage)
                fcn16s.load_state_dict(base['model_state_dict'])    
                model.copy_params_from_fcn16s(fcn16s)
            else:
                return model, start_epoch, start_iteration
    else:
        raise 'Model {} not available'.format(name)

    return model, start_epoch, start_iteration
