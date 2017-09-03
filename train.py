#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q1: 优化好像没有进行，完全没动，
solved: trainer的__init__(size_average=True)的问题，改成False就好了。

Created on Thu Aug 10 15:57:43 2017

@author: jjcao
"""
import argparse
from datasets import get_dataset
from datasets import transforms
from models import get_model
import torch
#from torchvision import transforms

import os
import os.path as osp
import datetime
import pytz
import yaml

here = osp.dirname(osp.abspath(__file__))
configurations = {
    # same configuration as original work
    # https://github.com/shelhamer/fcn.berkeleyvision.org
    1: dict( # for test
        max_iteration=100000,
        lr=1.0e-10, # learning rate
        momentum=0.99,
        weight_decay=0.0005,
        interval_validate=40,
        batch_size = 1,
        num_workers = 4,
    ),
    2: dict( # for fcn32s
        max_iteration=100000,
        lr=1.0e-10, # learning rate
        momentum=0.99,
        weight_decay=0.0005,
        interval_validate=4000,
        batch_size = 1,
        num_workers = 4,
    ),        
    3: dict( # for fcn16s
        max_iteration=100000,
        lr=1.0e-12,
        momentum=0.99,
        weight_decay=0.0005,
        interval_validate=4000,
        batch_size = 1,
        num_workers = 4,
        fcn32s_pretrained_model='?.pth.tar',
    ),
        
    4: dict( # for fcn8s
        max_iteration=100000,
        lr=1.0e-14,
        momentum=0.99,
        weight_decay=0.0005,
        interval_validate=4000,
        batch_size = 1,
        num_workers = 4,
        fcn16s_pretrained_model='?.pth.tar',
    ),
    5: dict( # for linknet
        max_iteration=100000,
        lr=5.0e-4, # learning rate
        lrd=5.0e-1, #learningRateDecay
        lrde= 10, #lrDecayEvery (default 100) Decay learning rate every X epoch by 1e-1
        interval_validate=40,
        batch_size = 4, # default 8
        momentum=0.99,
        weight_decay=2e-4,
        
        num_workers = 4,
    ),
}


def get_log_dir(model_name, config_id, cfg):
    # load config
    name = 'MODEL-%s_CFG-%03d' % (model_name, config_id)
    for k, v in cfg.items():
        v = str(v)
        if '/' in v:
            continue
        name += '_%s-%s' % (k.upper(), v)
    now = datetime.datetime.now(pytz.timezone('Asia/Shanghai'))
    # name += '_VCS-%s' % git_hash() # git_hash() need install command line tool or x-code?
    name += '_TIME-%s' % now.strftime('%Y%m%d-%H%M%S')
    # create out
    log_dir = osp.join(here, 'logs', name)
    if not osp.exists(log_dir):
        os.makedirs(log_dir)
    with open(osp.join(log_dir, 'config.yaml'), 'w') as f:
        yaml.safe_dump(cfg, f, default_flow_style=False)
    return log_dir

def get_optimizer(name):
    return {
        'fcn8s': torch.optim.SGD,
        'fcn16s': torch.optim.SGD,
        'fcn32s': torch.optim.SGD,
        'linknet': torch.optim.RMSprop,
    }[name]
    
def train(args):
    ##########################################
    # 0. preparation
    ##########################################
    # '0' for GPU 0; '0,2' for GPUs 0 and 2, etc
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    cuda = torch.cuda.is_available();
    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)
        
    cfg = configurations[args.config]
    log_dir = get_log_dir(args.arch, args.config, cfg)
    
    ##########################################
    # 1. dataset and dataloader    
    ##########################################
    dataset_dir = '/Users/jjcao/Documents/jjcao_data/VOCdevkit/VOC2012/'
    Dataset = get_dataset(args.dataset)
    
    # 
#    data_transform = transforms.Compose(
#            [transforms.Normalize(Dataset.mean_bgr), 
#            transforms.ToTensor() ]) # used by FCN   
    data_transform = transforms.Compose(
            [transforms.Normalize(Dataset.mean_bgr), 
             transforms.Rescale((args.im_rows, args.im_cols)),
             transforms.ToTensor() ]) 

    dataset = Dataset(dataset_dir=dataset_dir, split='train', transform=data_transform) 
    
    kwargs = {'num_workers': cfg['num_workers'], 'pin_memory': True} if cuda else {'num_workers': cfg['num_workers']}  
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=cfg['batch_size'], 
                                               shuffle=True, **kwargs)
    dataset = Dataset(dataset_dir=dataset_dir, split='train', transform=data_transform)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=cfg['batch_size'],
                                             shuffle=False, **kwargs)
    if __debug__:
        print("batch size is {}, length of train_loader is {}".
                                      format(cfg['batch_size'], len(train_loader)))
        im, lbl = dataset[0]
        print(im.shape, lbl.shape)

    ##########################################
    # 2. model
    ##########################################
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(args.resume)
        
    model, start_epoch, start_iteration = get_model(args.arch, 
                                                    len(Dataset.class_names),
                                                    checkpoint, cfg)
    if cuda:
        model = model.cuda()        
#    if __debug__:
#        print("Model: {}. Training begin at {}".format(args.resume))
#    return 

    ##########################################
    # 3. optimizer
    ##########################################
    #import pdb; pdb.set_trace()
    Optimizer = get_optimizer(args.arch)
    optim = Optimizer(model.parameters(), lr=cfg['lr'], 
                                momentum=cfg['momentum'], weight_decay=cfg['weight_decay'])
    
    if args.resume:
        optim.load_state_dict(checkpoint['optim_state_dict'])

        
    ########################################## 
    # 4. train  
    ##########################################
    cuda = torch.cuda.is_available()
    from trainer import Trainer      
    trainer = Trainer(
        cuda=cuda,
        model=model,
        optimizer=optim,
        train_loader=train_loader,
        val_loader=val_loader,
        out=log_dir,
        max_iter=cfg['max_iteration'],
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
    
    parser.add_argument('--dataset', nargs='?', type=str, default='pascal', 
                        help='Dataset to use [\'pascal, camvid, ade20k etc\']')
     
    parser.add_argument('--arch', nargs='?', type=str, default='linknet', 
                        help='Architecture to use [\'fcn8s, unet, segnet, linknet, pspnet etc\']')

    parser.add_argument('-c', '--config', type=int, default=5,
                        choices=configurations.keys())
    
    parser.add_argument('-g', '--gpu', type=str, default='0')
  
    parser.add_argument('--resume', help='Checkpoint path')
#    parser.add_argument('--resume', help='Checkpoint path', type=str, 
#          default='./logs/MODEL-fcn32s_CFG-001_MAX_ITERATION-100000_BATCH_SIZE-1_NUM_WORKERS-4_LR-1e-10_MOMENTUM-0.99_WEIGHT_DECAY-0.0005_INTERVAL_VALIDATE-40_TIME-20170831-161727/checkpoint.pth.tar')
            
    parser.add_argument('--im_rows', nargs='?', type=int, default=256, 
                        help='Height of the input image')
    parser.add_argument('--im_cols', nargs='?', type=int, default=256, 
                        help='Height of the input image')
    
    parser.add_argument('--feature_scale', nargs='?', type=int, default=1, 
                        help='Divider for # of features to use') 
    
    args = parser.parse_args()
    
    train(args)