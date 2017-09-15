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
import datetime
import pytz
import yaml

import torch

from datasets import get_dataset
from datasets import transforms

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

def get_log_dir(model_name, cfg):
    # load config
    name = model_name
    for k, v in cfg.items():
        v = str(v)
        if '/' in v:
            continue
        name += '_%s-%s' % (k.upper(), v)
    now = datetime.datetime.now(pytz.timezone('Asia/Shanghai'))
    # name += '_VCS-%s' % git_hash() # git_hash() need install command line tool or x-code?
    name += '_TIME-%s' % now.strftime('%Y%m%d-%H%M%S')
    # create out
    here = osp.dirname(osp.abspath(__file__))
    log_dir = osp.join(here, 'logs', name)
    if not osp.exists(log_dir):
        os.makedirs(log_dir)
    with open(osp.join(log_dir, 'config.yaml'), 'w') as f:
        yaml.safe_dump(cfg, f, default_flow_style=False)
    return log_dir

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
    cuda = torch.cuda.is_available();
    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)
        print("cuda devices: {} are ready".format(args['gpu']))
        
        
    log_dir = get_log_dir(args['model'], cfg)
    
    
    ##########################################
    # 1. dataset and dataloader    
    ##########################################
    dataset_dir = args['dataset_dir']
    Dataset = get_dataset(args['dataset'])
    
    if 'im_rows' in args:
        data_transform = transforms.Compose(
                [transforms.Normalize(Dataset.mean_bgr), 
                 transforms.Rescale( ( int(args['im_rows']), int(args['im_cols']) ) ),
                 transforms.ToTensor() ]) 
    else:     
        data_transform = transforms.Compose(
                [transforms.Normalize(Dataset.mean_bgr), 
                transforms.ToTensor() ]) # used by FCN   

    
    dataset = Dataset(dataset_dir=dataset_dir, split='train', transform=data_transform) 
    
    batch_size = int(args['batch_size'])
    num_workers = int(args['num_workers'])
    # does num_workers work if CPU?
    # kwargs = {'num_workers': cfg['num_workers'], 'pin_memory': True} if cuda else {} 
    kwargs = {'num_workers': num_workers, 'pin_memory': True} if cuda else {'num_workers': num_workers}  
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                               shuffle=True, **kwargs)
    dataset = Dataset(dataset_dir=dataset_dir, split='val', transform=data_transform)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=False, **kwargs)
    if __debug__:
        print("batch size is {}, length of train_loader is {}".
                                      format(batch_size, len(train_loader)))
        im, lbl = dataset[0]
        print(im.shape, lbl.shape)


    ##########################################
    # 2. model
    ##########################################
    
    from models import get_model
    checkpoint = None
    #import pdb; pdb.set_trace()
    if ('checkpoint_dir' in args) and (len(args['checkpoint_dir'])>0):
        checkpoint = torch.load(args['checkpoint_dir'])
        
    model, start_epoch, start_iteration = get_model(args['model'], 
                                                    len(Dataset.class_names),
                                                    checkpoint, args)
    if cuda:
        model = model.cuda()        
#    if __debug__:
#        print("Model: {}. Training begin at {}".format(args.resume))
#    return 


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
        train_loader=train_loader,
        val_loader=val_loader,
        out=log_dir,
        max_iter=cfg['max_iter'],
        l_rate = cfg['lr'],
        l_rate_decay = cfg.get('lrd', 1.0),
        interval_validate=cfg.get('interval_validate', len(train_loader)),
    )
        
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    trainer.train()

        
if __name__ == '__main__':
    #torch.set_num_threads(1)
    
    parser = argparse.ArgumentParser(description='Hyperparams')    
    parser.add_argument('-c', '--config', type=str, default='config_hed.ini') 
    args = parser.parse_args()
    
    train(args)