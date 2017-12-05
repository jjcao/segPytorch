#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 23:56:44 2017

@author: jjcao
"""

from PIL import Image    
import numpy as np
import os
import torch

dataset_dir = '/Users/jjcao/Documents/data/chen/'
img_file = os.path.join(dataset_dir, '134_50_134_462_65_166_209_166.png')
img = Image.open(img_file).convert('RGB')
im_fine_size = [20,20]

img = img.resize((im_fine_size[0]*2, im_fine_size[1]))

ima = np.array(img)

imt = torch.from_numpy(ima).float()

