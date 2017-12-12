#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 14:10:42 2017

@author: jjcao
"""
import os
import torch
import sys
import pdb

print('__Python Version:', sys.version)
print('pyTorch Version:', torch.__version__)
print('cuda version:')
from subprocess import call
#nvcc --version
print('__CUDNN Version:', torch.backends.cudnn.version())

# it is number of avaliable gpu in hardware, not changed after CUDA_VISIBLE_DEVICES
print('__Number CUDA Devices:', torch.cuda.device_count()) 
print('__Current CUDA Devices:', torch.cuda.current_device())

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

pdb.set_trace()
#torch.cuda.set_device(1)
print(torch.cuda.get_device_name(1))

print('after CUDA visible')
print('__Number CUDA Devices:', torch.cuda.device_count())
print('__Current CUDA Devices:', torch.cuda.current_device())
