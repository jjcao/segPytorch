#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 15:57:43 2017

@author: jjcao
"""
import argparse
from datasets import get_dataset
from datasets.util import Rescale, Normalize, ToTensor
from models import get_model
import torch
from torchvision import transforms

import os
import os.path as osp
import shlex
import subprocess
import datetime
import pytz
import yaml

import visdom

here = osp.dirname(osp.abspath(__file__))
configurations = {
    # same configuration as original work
    # https://github.com/shelhamer/fcn.berkeleyvision.org
    1: dict(
        max_iteration=100000,
        lr=1.0e-10, # learning rate
        momentum=0.99,
        weight_decay=0.0005,
        interval_validate=4000,
    )
}

def git_hash():
    cmd = 'git log -n 1 --pretty="%h"'
    hash = subprocess.check_output(shlex.split(cmd)).strip()
    return hash

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


def train(args):
    # 0. preparation
    # '0' for GPU 0; '0,2' for GPUs 0 and 2, etc
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    cuda = torch.cuda.is_available();
    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)
        
    cfg = configurations[args.config]
    log_dir = get_log_dir(args.arch, args.config, cfg)
    
    # 1. dataset and dataloader    
    dataset_dir = '/Users/jjcao/Documents/jjcao_data/VOCdevkit/VOC2012/'
    Dataset = get_dataset(args.dataset)
    
    # 
#    data_transform = transforms.Compose([Rescale(256), RandomCrop(224), 
#                                   Normalize(), ToTensor() ])
    data_transform = transforms.Compose([Normalize(), ToTensor() ]) # used by FCN

    dataset = Dataset(dataset_dir=dataset_dir, split='train', transform=data_transform)    
    kwargs = {'pin_memory': True} if cuda else {}  
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, 
                                             shuffle=True, num_workers=4, **kwargs)
    dataset = Dataset(dataset_dir=dataset_dir, split='val', transform=data_transform)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, 
                                             shuffle=False, **kwargs)
#    test_image, test_segmap = dataset[0]
#    if cuda:     
#        test_image = Variable(test_image.unsqueeze(0).cuda(0))
#    else:
#        test_image = Variable(test_image.unsqueeze(0))


    # Setup visdom for visualization
#    vis = visdom.Visdom()
#
#    loss_window = vis.line(X=torch.zeros((1,)).cpu(),
#                           Y=torch.zeros((1)).cpu(),
#                           opts=dict(xlabel='minibatches',
#                                     ylabel='Loss',
#                                     title='Training Loss',
#                                     legend=['Loss']))
    
    
    # 2. model
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(args.resume)
        
    model, start_epoch,  start_iteration= get_model(args.arch, dataset.n_classes, checkpoint)
    if cuda:
        model = model.cuda()        
    
    # 3. optimizer  
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg['lr'], 
                                momentum=cfg['momentum'], weight_decay=cfg['weight_decay'])
    if args.resume:
        optimizer.load_state_dict(checkpoint['optim_state_dict'])
 
    
     
    # 4. train        
    trainer = Trainer(
        cuda=cuda,
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        train_loader=train_loader,
        out=log_dir,
        max_iter=cfg['max_iteration'],
        interval_validate=cfg.get('interval_validate', len(train_loader)),
    )
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    trainer.train()

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    
    parser.add_argument('--dataset', nargs='?', type=str, default='pascal', 
                        help='Dataset to use [\'pascal, camvid, ade20k etc\']')
     
    parser.add_argument('--arch', nargs='?', type=str, default='fcn32s', 
                        help='Architecture to use [\'fcn8s, unet, segnet etc\']')

    parser.add_argument('-c', '--config', type=int, default=1,
                        choices=configurations.keys())
    
    parser.add_argument('-g', '--gpu', type=str, default='0')
    parser.add_argument('--resume', help='Checkpoint path')
    
    parser.add_argument('--batch_size', nargs='?', type=int, default=1, 
                        help='Batch Size')
    
    
    parser.add_argument('--img_rows', nargs='?', type=int, default=256, 
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=256, 
                        help='Height of the input image')
    
    parser.add_argument('--feature_scale', nargs='?', type=int, default=1, 
                        help='Divider for # of features to use') 
    
    args = parser.parse_args()
    
    train(args)