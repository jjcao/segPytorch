from PIL import Image,ImageStat
import os
L1=[]
L2=[]
L3=[]
for filename in os.listdir(r"D:\\sc_2\\data_processing_exercise\\raw\\Origin"):
    filepath=os.path.join("D:\\sc_2\\data_processing_exercise\\raw\\Origin",filename)
    im=Image.open(filepath)
    r,g,b=im.split()
    stat=ImageStat.Stat(im)
    L1.append(stat.mean[0])
    L2.append(stat.mean[1])
    L3.append(stat.mean[2])
print sum(L1)/len(L1)
print sum(L2)/len(L2)
print sum(L3)/len(L3)
   #print stat.mean
