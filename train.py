#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q1: 优化好像没有进行，完全没动，
solved: trainer的__init__(size_average=True)的问题，改成False就好了。

Created on Thu Aug 10 15:57:43 2017

@author: jjcao
"""
#import pdb; pdb.set_trace()
import argparse
import configparser

import os
import os.path as osp

import torch

from datasets import get_dataset
from datasets import transforms
import utils.log as log

import ast

def read_cfg(cfg_dir):
    cfgp = configparser.ConfigParser()
    cfgp.read(cfg_dir)
    #cfgp.sections()
    cfg = cfgp['CONFIG']
    args = cfgp['ARGUMENT']
    
    args = dict( zip(args.keys(), args.values()) )  
    
    val = []
    for v in cfg.values():
        val.append(ast.literal_eval(v))
        
    cfg = dict( zip(cfg.keys(), val) ) 
    return args, cfg

def get_optimizer(name):
    return {
        'SGD': torch.optim.SGD,
        'RMSprop': torch.optim.RMSprop,
    }[name]
 
    
    
def train(args):
    ##########################################
    # 0. preparation
    ##########################################
    args, cfg = read_cfg(args.config)
        
    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
    cuda = torch.cuda.is_available()
    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)
        print("cuda devices: {} are ready".format(args['gpu']))
        
    here = osp.dirname(osp.abspath(__file__))    
    log_dir = log.get_log_dir(here, args['model'], cfg)
    
    
    ##########################################
    # 1. dataset and dataloader    
    ##########################################
    dataset_dir = args['dataset_dir']
    Datasets = {'train': get_dataset(args['dataset_train']),
                'val': get_dataset(args['dataset_val'])}
    
    if 'im_rows' in args:
        data_transforms = {
        'train': transforms.Compose(
                [transforms.Normalize(Datasets['train'].mean_bgr), 
                 transforms.Rescale( ( int(args['im_rows']), int(args['im_cols']) ) ),
                 transforms.ToTensor() ]),
        'val': transforms.Compose(
                [transforms.Normalize(Datasets['train'].mean_bgr), 
                 transforms.Rescale( ( int(args['im_rows']), int(args['im_cols']) ) ),
                 transforms.ToTensor() ]),
        }
    else:  # used by FCN  
        data_transforms = {
        'train': transforms.Compose(
                [transforms.Normalize(Datasets['train'].mean_bgr), 
                transforms.ToTensor() ]),
        'val': transforms.Compose(
                [transforms.Normalize(Datasets['train'].mean_bgr), 
                transforms.ToTensor() ]),
        }
    
    datasets = {x: Datasets[x](root=dataset_dir, transform=data_transforms[x]) 
                  for x in ['train', 'val']} 

    shuffle = {'train': True, 'val':False}
    batch_size = int(args['batch_size'])
    num_workers = int(args['num_workers'])
    # does num_workers work if CPU?
    # kwargs = {'num_workers': cfg['num_workers'], 'pin_memory': True} if cuda else {} 
    kwargs = {'num_workers': num_workers, 'pin_memory': True} if cuda else {'num_workers': num_workers}      
    dataloders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=batch_size,
                                             shuffle=shuffle[x], **kwargs)
              for x in ['train', 'val']}

    if __debug__:
        print("batch size is {}, length of train_loader is {}".
              format(batch_size, len(dataloders['train'])))
        im, lbl, _ = datasets['train'][0]
        print(im.shape, lbl.shape)


    ##########################################
    # 2. model
    ##########################################
    
    from models import get_model
    checkpoint = None
    if ('checkpoint_dir' in args) and (len(args['checkpoint_dir'])>0):
        if cuda:
            checkpoint = torch.load(args['checkpoint_dir']) 
        else:
            checkpoint = torch.load(args['checkpoint_dir'], map_location=lambda storage, loc: storage)
        
    model, start_epoch, start_iteration = get_model(args['model'], 
                                                    len(datasets['train'].class_names),
                                                    checkpoint, args)
    if cuda:
        model = model.cuda()        


    ##########################################
    # 3. optimizer
    ##########################################
    
    Optimizer = get_optimizer(args['optimizer'])
    optim = Optimizer(model.parameters(), lr=cfg['lr'], 
                                momentum=cfg['momentum'], weight_decay=cfg['weight_decay'])
    
    if ('checkpoint_dir' in args) and (len(args['checkpoint_dir'])>0):
        optim.load_state_dict(checkpoint['optim_state_dict'])

        
    ########################################## 
    # 4. train  
    ##########################################
       
    from trainers import get_trainer
    Trainer = get_trainer(args['trainer']) 
    trainer = Trainer(
        cuda=cuda,
        model=model,
        optimizer=optim,
        train_loader=dataloders['train'],
        val_loader=dataloders['val'],
        out=log_dir,
        max_iter=cfg['max_iter'],
        l_rate = cfg['lr'],
        l_rate_decay = cfg.get('lrd', 1.0),
        interval_validate=cfg.get('interval_validate', len(dataloders['train'])),
    )
        
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    log_step = 1
    if ('log_step' in args):
        log_step = args['log_step']
    trainer.train(log_step)

        
if __name__ == '__main__':
    #torch.set_num_threads(1)
    
    parser = argparse.ArgumentParser(description='Hyperparams')    
    parser.add_argument('-c', '--config', type=str, default='config_fcn32s_sc2.ini') 
    args = parser.parse_args()
    
    train(args)