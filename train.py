#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 15:57:43 2017

@author: jjcao
"""
import argparse
from datasets import get_dataset
from datasets.util import show_sample_batch, Rescale, RandomCrop, Normalize, ToTensor
from models import get_model
import torch
from torchvision import transforms

def train(args):
    # 1. dataset and dataloader    
    dataset_dir = '/Users/jjcao/Documents/jjcao_data/VOCdevkit/VOC2012/'
    Dataset = get_dataset(args.dataset)
    
    data_transform = transforms.Compose([Rescale(256),
                                   RandomCrop(224), 
                                   Normalize(),
                                   ToTensor() ])

    dataset = Dataset(dataset_dir=dataset_dir, split='train', transform=data_transform)    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, 
                                             shuffle=True, num_workers=4)

    # 2. model
    model = get_model(args.arch, dataset.n_classes)
#    model = torchfcn.models.FCN32s(n_class=21)
#    start_epoch = 0
#    start_iteration = 0
#    if resume:
#        checkpoint = torch.load(resume)
#        model.load_state_dict(checkpoint['model_state_dict'])
#        start_epoch = checkpoint['epoch']
#        start_iteration = checkpoint['iteration']
#    else:
#        vgg16 = torchfcn.models.VGG16(pretrained=True)
#        model.copy_params_from_vgg16(vgg16)


    # 3. optimizer    
  
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['image'].size(),
                       sample_batched['label'].size())

        #observe 4th batch and stop.
        show_sample_batch(i_batch, sample_batched) 
        if i_batch == 2:        
            break
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    
    parser.add_argument('--dataset', nargs='?', type=str, default='pascal', 
                        help='Dataset to use [\'pascal, camvid, ade20k etc\']')
     
    parser.add_argument('--arch', nargs='?', type=str, default='fcn32s', 
                        help='Architecture to use [\'fcn8s, unet, segnet etc\']')

    parser.add_argument('--img_rows', nargs='?', type=int, default=256, 
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=256, 
                        help='Height of the input image')
    parser.add_argument('--n_epoch', nargs='?', type=int, default=100, 
                        help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=1, 
                        help='Batch Size')
    parser.add_argument('--l_rate', nargs='?', type=float, default=1e-5, 
                        help='Learning Rate')
    parser.add_argument('--feature_scale', nargs='?', type=int, default=1, 
                        help='Divider for # of features to use') 
    
    args = parser.parse_args()
    
    train(args)