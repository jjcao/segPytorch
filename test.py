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
import scipy.misc as misc
import torch
from torch.autograd import Variable
import tqdm

from datasets.test_dataset import TestDataset
from datasets import transforms

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='Model path', default='../../output/fcn32s_model_best_nclass_21.pth.tar')
    parser.add_argument('-i', '--input', help='path of images', type=str, default='../../input/')#/Users/jjcao/Documents/data/    
    parser.add_argument('-o', '--output', help='output path', type=str, default='../../output')  
    parser.add_argument('-g', '--gpu', help='-1 for cpu', type=int, default=-1)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)  
    cuda = torch.cuda.is_available();
    model_file = args.model

    ############################## 
    ### data
    ############################## 
    # sometimes, we need set mean_bgr
    bname = osp.basename(model_file)
    bname = osp.splitext(bname)[0]
    bname = osp.splitext(bname)[0]
    id = bname.index('nclass') 
    id1 = bname.index('_', id)
    n_class = int(bname[id1+1:len(bname)])
    
    data_transform = transforms.Compose(
                [transforms.Normalize(TestDataset.mean_bgr), 
                transforms.ToTensor() ]) # used by FCN   
    
    
    dataset = TestDataset(args.input, data_transform, n_class)
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {'num_workers': 1}
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                             shuffle=False, **kwargs)    
    im, lbl, name = dataset[0]
    print(im.shape, lbl.shape, name)
        
    ############################## 
    ### model
    ############################## 
    if bname.startswith('fcn32s'):
        model_type = 'fcn32s'
    elif bname.startswith('fcn16s'):
        model_type = 'fcn16s'
    elif bname.startswith('fcn8s'):
        if bname.startswith('fcn8s-atonce'):
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
    print('==> testing')
    for batch_idx, (data, target, names) in tqdm.tqdm(enumerate(val_loader),
                                               total=len(val_loader),
                                               ncols=80, leave=False):
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        scores = model(data)#outputs = model(images)
                
        preds = scores.data.max(1)[1].cpu().numpy()[:, :, :]#pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=1)
            
        for name, pred in zip(names,preds):
            decoded = dataset.decode_segmap(pred)
            print(np.unique(pred))
            output = osp.join(args.output, '%s.png'%name)
            misc.imsave(output, decoded)
            #skimage.io.imsave(output, decoded)
            print("Segmentation Mask Saved at: {}".format(args.output))

if __name__ == '__main__':
    main()
