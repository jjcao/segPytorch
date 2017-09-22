#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 10:35:54 2017

steal lots from https://github.com/wkentaro/pytorch-fcn/

@author: jjcao
"""

#!/usr/bin/env python
#import pdb; pdb.set_trace()
import argparse
import os
import os.path as osp

import numpy as np
import skimage.io
import torch
from torch.autograd import Variable
import tqdm

from datasets import get_dataset
from datasets import transforms
from utils import utils

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='Model path', default='../../output/fcn32s_model_best.pth.tar')
    parser.add_argument('-d', '--data', type=str, default='../../data/')#/Users/jjcao/Documents/data/
    parser.add_argument('-g', '--gpu', type=int, default=-1)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)  
    cuda = torch.cuda.is_available();
    model_file = args.model

    ############################## 
    ### data
    ############################## 
    ValDataset = get_dataset('VOC2011')
    data_transform = transforms.Compose(
                [transforms.Normalize(ValDataset.mean_bgr), 
                transforms.ToTensor() ]) # used by FCN   
    
    
    dataset = ValDataset(root=args.data, transform=data_transform)
    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {'num_workers': 4}
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                             shuffle=False, **kwargs)    

    n_class = len(val_loader.dataset.class_names)
  
        
    ############################## 
    ### model
    ############################## 
    if osp.basename(model_file).startswith('fcn32s'):
        model_type = 'fcn32s'
    elif osp.basename(model_file).startswith('fcn16s'):
        model_type = 'fcn16s'
    elif osp.basename(model_file).startswith('fcn8s'):
        if osp.basename(model_file).startswith('fcn8s-atonce'):
            model_type = 'fcn8s-atonce'
        else:
            model_type = 'fcn8s'
    else:
        raise ValueError
        
    from models import get_model    
    if cuda:
        checkpoint = torch.load(model_file) 
    else:
        checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        
    model, start_epoch, start_iteration = get_model(model_type, n_class, checkpoint, args)

    
    if cuda:
        model = model.cuda()
  
    print('==> Loading %s model file: %s, at iteration: %d' % 
          (model.__class__.__name__, model_file, start_iteration)) 
    
    model.eval()


    ############################## 
    ### Evaluating
    ############################## 
    print('==> Evaluating with VOC2011ClassSeg seg11valid')
    visualizations = []
    label_trues, label_preds = [], []
    for batch_idx, (data, target) in tqdm.tqdm(enumerate(val_loader),
                                               total=len(val_loader),
                                               ncols=80, leave=False):
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        score = model(data)

        imgs = data.data.cpu()
        lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
        lbl_true = target.data.cpu()
        for img, lt, lp in zip(imgs, lbl_true, lbl_pred):
            img, lt = val_loader.dataset.untransform(img, lt)
            label_trues.append(lt)
            label_preds.append(lp)
            if len(visualizations) < 9:
                #import pdb; pdb.set_trace()
                viz = utils.visualize_segmentation(
                    lbl_pred=lp, lbl_true=lt, img=img, n_class=n_class)#,
                    #label_names=val_loader.dataset.class_names)
                visualizations.append(viz)
    metrics = utils.label_accuracy_score(label_trues, label_preds, n_class=n_class)
    metrics = np.array(metrics)
    metrics *= 100
    print('''\
Accuracy: {0}
Accuracy Class: {1}
Mean IU: {2}
FWAV Accuracy: {3}'''.format(*metrics))

    viz = utils.get_tile_image(visualizations)
    skimage.io.imsave('viz_evaluate.png', viz)


if __name__ == '__main__':
    main()
