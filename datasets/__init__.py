#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 17:07:52 2017

@author: jjcao
"""
from datasets.voc_dataset import VOC2011ClassSeg
from datasets.voc_dataset import VOC2012ClassSeg
from datasets.voc_dataset import SBDClassSeg
from datasets.sps_dataset import Sc2ClassSeg
from datasets.sps_dataset import ScValClassSeg
from datasets.sps_dataset import ScTestClassSeg
    
#import transforms

#__all__ = ['transforms']

def get_dataset(name):
    """get_loader
    :param name:
    """
    return {
        'VOC2011': VOC2011ClassSeg,
        'VOC2012ClassSeg': VOC2012ClassSeg,
        'SBD': SBDClassSeg,
        'SC2': Sc2ClassSeg,
        'SCVAL': ScValClassSeg,
        'SCTEST': ScTestClassSeg,
    }[name]
 
    
