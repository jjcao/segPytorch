import os
#import pdb
#pdb.set_trace()
import numpy as np
#import skimage,skimage.io
from skimage import io,transform
import operator
import argparse
import configparser
import ast
def read_cfg(cfg_dir):
    
    cfgp = configparser.ConfigParser()
    cfgp.read(cfg_dir)
    args = cfgp['ARGUMENT']
    cfg = cfgp['CONFIG']
    args = dict(zip(args.keys(), args.values()))
    val = []
    for v in cfg.values():
        val.append(ast.literal_eval(v))
        
    cfg = dict(zip(cfg.keys(), val))   
    return args,cfg
def Cut(args):
    args,cfg= read_cfg(args.config)
    
    imgname=os.listdir(args['imgname_dir'])
    len1=len(imgname)
    M=np.loadtxt(args['coord_dir'],delimiter=',',dtype=np.str)
    name=M[:,0]
    len2=len(name)
    for i in range(0,len1):
        inde=-1
        for j in range(0,len2):
        
            if operator.__eq__(imgname[i],name[j]):
                inde=j
        if operator.__eq__(inde,-1):
            print(imgname[i]+',is no fount in 3p.txt!')
            continue
        img=io.imread(args['imgname_dir']+imgname[i])
    
        xmin=min(int(M[inde][1]),int(M[inde][3]),int(M[inde][5]),int(M[inde][7]))
        if xmin<0:
            xmin=0
   
        ymin=min(int(M[inde][2]),int(M[inde][4]),int(M[inde][6]),int(M[inde][8]))
        if ymin<0:
            ymin=0
   
        wid=abs(int(M[inde][5])-int(M[inde][7]))
        h=abs(int(M[inde][2])-int(M[inde][4]))
        #cut
        cut_new=img[ymin:ymin+h+1,xmin:xmin+wid+1]
        io.imsave(args['new_cut_dir']+'/'+np.str(i+1)+'.png',cut_new)

        #resize
        resize_new=transform.resize(cut_new,(cfg['wid_len'],cfg['h_len']))
        io.imsave(args['new_resize_dir']+'/'+np.str(i+1)+'.png',resize_new)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Hyperparams')    
    parser.add_argument('-c', '--config', type=str, default='cut.ini') 
    args = parser.parse_args()
    
    Cut(args)
        




#for imgname in os.listdir('/disk1/wang/data_processing_exercise/raw/Origin'):
    #print(imgname)
    #img=cv2.imread('/disk1/wang/data_processing_exercise/raw/Origin/'+imgname)
    #print(img.shape)
    #cv2.imshow(img)
