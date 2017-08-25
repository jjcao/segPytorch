#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test torchvision.transforms

Created on Thu Aug 24 22:25:04 2017

@author: jjcao
"""

import torchvision.transforms

from skimage import io


trans = transforms.Compose([transforms.CenterCrop(10), transforms.ToTensor()])