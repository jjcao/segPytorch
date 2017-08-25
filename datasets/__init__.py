#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 17:07:52 2017

@author: jjcao
"""
from datasets.voc_dataset import VocDataset

#__all__ = ['transforms']

def get_dataset(name):
    """get_loader
    :param name:
    """
    return {
        'pascal': VocDataset,
        #'camvid': camvidLoader,
    }[name]
 
    
