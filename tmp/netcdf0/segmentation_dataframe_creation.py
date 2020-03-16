import numpy as np
import netCDF4 as nc
import scipy as scp
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import seaborn as sns
from functions import *

#img         = np.load('../data/img.npy')
img_fp = '/home/marcosrdac/projects/oil_spill/netcdf/S1A_IW_SLC__1SDV_20181008T052755_20181008T052822_024040_02A081_3524_deb_Orb_ML_msk.nc'
img         = nc.Dataset(img_fp)['Intensity_VV_db']
#img_avg3x3     = moving_mean(img, 3)
bg_sea_msk  = np.load('../data/bg_sea_msk.npy')
fg_spot_msk = np.load('../data/fg_spot_msk.npy')

total_labeled = np.sum(bg_sea_msk) + np.sum(fg_spot_msk)

shape = img.shape

x      = []
y      = []
value  = []
avg3x3 = []
std3x3 = []
label  = [] # 0 bg_sea, 1 fg_spot

k=0
for yi in range(    2, img.shape[0]-2, 3):
    for xi in range(2, img.shape[1]-2, 3):
        labeled = 0
        if bg_sea_msk[yi,xi]:
            labeli  = 0
            labeled = 1
        elif fg_spot_msk[yi,xi]:
            labeli  = 1
            labeled = 1
        if labeled == 1:
            x.append(xi)
            y.append(yi)
            #value.append(img[yi,xi])
            #avg3x3.append(img_avg3x3[yi,xi])
            std3x3.append(np.std(img[ yi-1:yi+2,
                                      xi-1:xi+2 ]))
            label.append(labeli)
            print(100*k/total_labeled*9)
            k+=1

#np.save('x',np.array(x));           del x
#np.save('y',np.array(y));           del y
#np.save('value',np.array(value)); del value
#np.save('avg',np.array(avg));     del avg
#np.save('std',np.array(std));     del std
#np.save('label',np.array(label)); del label

np.save('x',np.array(x));             del x
np.save('y',np.array(y));             del y
#np.save('value',np.array(value));    del value
#np.save('avg3x3',np.array(avg3x3));  del avg3x3
np.save('std3x3',np.array(std3x3));   del std3x3
np.save('label',np.array(label));     del label
