#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 10:40:40 2017

@author: jjcao
"""            
import configparser

def write():
    cfg = configparser.ConfigParser()
    cfg['DEFAULT'] = {'max_iteration':'100000',
       'lr':'1.0e-10', # learning rate
       }
    
    with open('example.ini', 'w') as cfgfile:
        cfg.write(cfgfile)
        
def read_cfg(cfg_dir):
    cfgp = configparser.ConfigParser()
    cfgp.read(cfg_dir)
    #cfgp.sections()
    cfg = cfgp['CONFIG']
    args = cfgp['ARGUMENT']
    return args, cfg
    
    
args, cfg = read_cfg('config_fcn32s_cpu.ini')
args = dict( zip(args.keys(), args.values()) )    
cfg = dict( zip(cfg.keys(), cfg.values()) )  
print('args')
for k, v in args.items():
    print("[{}, {}]".format(k, v))
    
print('cfg')    
val = []
for v in cfg.values():
    val.append(float(v))
for k, v in cfg.items():
    print("[{}, {}]".format(k, v))
 

    
'im_rows' in args    
#args['im_rows']



#configurations = {
#    # same configuration as original work
#    # https://github.com/shelhamer/fcn.berkeleyvision.org
#    1: dict( # for test
#        max_iteration=100000,
#        lr=1.0e-10, # learning rate
#        momentum=0.99,
#        weight_decay=0.0005,
#        interval_validate=40,
#        batch_size = 1,
#        num_workers = 4,
#    ),
#    2: dict( # for fcn32s
#        max_iteration=100000,
#        lr=1.0e-10, # learning rate
#        momentum=0.99,
#        weight_decay=0.0005,
#        interval_validate=4000,
#        batch_size = 1,
#        num_workers = 4,
#    ),        
#    3: dict( # for fcn16s
#        max_iteration=100000,
#        lr=1.0e-12,
#        momentum=0.99,
#        weight_decay=0.0005,
#        interval_validate=4000,
#        batch_size = 1,
#        num_workers = 4,
#        fcn32s_pretrained_model='?.pth.tar',
#    ),
#        
#    4: dict( # for fcn8s
#        max_iteration=100000,
#        lr=1.0e-14,
#        momentum=0.99,
#        weight_decay=0.0005,
#        interval_validate=4000,
#        batch_size = 1,
#        num_workers = 4,
#        fcn16s_pretrained_model='?.pth.tar',
#    ),
#    5: dict( # for linknet
#        max_iteration=100000,
#        lr=5.0e-4, # learning rate
#        lrd=5.0e-1, #learningRateDecay
#        lrde= 10, #lrDecayEvery (default 100) Decay learning rate every X epoch by 1e-1
#        interval_validate=40,
#        batch_size = 4, # default 8
#        momentum=0.99,
#        weight_decay=2e-4,
#        
#        num_workers = 4,
#    ),
#}