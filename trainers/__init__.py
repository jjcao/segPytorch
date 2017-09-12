#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:08:05 2017

@author: jjcao
"""

from trainers.trainer import Trainer
from trainers.trainer_hed import TrainerHed

def get_trainer(name):
    return {
        'default': Trainer,
        'trainerHed': TrainerHed,
    }[name]