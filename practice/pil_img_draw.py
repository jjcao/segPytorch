#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 17:52:56 2017

@author: jjcao
"""
from PIL import Image, ImageDraw    
import numpy as np
import os

def naive_line(r0, c0, r1, c1):
    # The algorithm below works fine if c1 >= c0 and c1-c0 >= abs(r1-r0).
    # If either of these cases are violated, do some switches.
    if abs(c1-c0) < abs(r1-r0):
        # Switch x and y, and switch again when returning.
        xx, yy, val = naive_line(c0, r0, c1, r1)
        return (yy, xx, val)

    # At this point we know that the distance in columns (x) is greater
    # than that in rows (y). Possibly one more switch if c0 > c1.
    if c0 > c1:
        return naive_line(r1, c1, r0, c0)

    # We write y as a function of x, because the slope is always <= 1
    # (in absolute value)
    x = np.arange(c0, c1+1, dtype=float)
    y = x * (r1-r0) / (c1-c0) + (c1*r0-c0*r1) / (c1-c0)

    valbot = np.floor(y)-y+1
    valtop = y-np.floor(y)

    return (np.concatenate((np.floor(y), np.floor(y)+1)).astype(int), np.concatenate((x,x)).astype(int),
            np.concatenate((valbot, valtop)))
    
def draw_2lines(im_arr, xy, im_size):
        
    x, y, val  = naive_line(*xy[0:4])
    x[x>im_size[0]-1]=im_size[0]-1
    x[x<0]=0
    y[y>im_size[1]-1]=im_size[1]-1
    y[y<0]=0
    
    im_arr[y, x, 0] = 255
    im_arr[y, x, 1] = 255
    im_arr[y, x, 2] = 0
    
    
    x, y, val  = naive_line(*xy[4:8])
    x[x>im_size[0]-1]=im_size[0]-1
    x[x<0]=0
    y[y>im_size[1]-1]=im_size[1]-1
    y[y<0]=0
    
    im_arr[y, x, 0] = 255
    im_arr[y, x, 1] = 0
    im_arr[y, x, 2] = 0
    
    
##################################    
    
dataset_dir = '/Users/jjcao/Documents/data/chen/'
img_file = os.path.join(dataset_dir, '134_50_134_462_65_166_209_166.png')
img = Image.open(img_file).convert('RGB')
im_in_size = [img.size[0], img.size[1]]
im_in_size[0] = im_in_size[0]*0.5
#im_fine_size = im_in_size
im_fine_size = [256,256]

img = img.resize((im_fine_size[0]*2, im_fine_size[1]))


s = '/'
im_name = img_file[img_file.rfind(s) + 1:-4] # a string
box  = im_name.split('_')
box = [float(x) for x in box]

box[::2] = [x / im_in_size[0] * im_fine_size[0] for x in box[::2]]
box[1::2] = [y / im_in_size[1] * im_fine_size[1] for y in box[1::2]]

#box = [x * im_fine_size for x in box]

box = np.asarray(box)
box[::2] = box[::2] + img.size[0]*0.5
#box = Box.tolist()

# solution 0: draw on numpy array
im = np.array(img)
draw_2lines(im, box, img.size)
im = Image.fromarray(im)
im.show()


# solution 1: draw on pil image
#draw = ImageDraw.Draw(img)    
#draw.line(box, fill='red')
#del draw
#
#img.show()


