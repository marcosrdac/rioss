import numpy as np
import netCDF4 as nc
import scipy as scp
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import seaborn as sns
from functions import *
from time import time, strftime, gmtime
from datetime import timedelta as convert
from numba import jit,prange


img_fp = '/home/marcosrdac/projects/oil_spill/netcdf/S1A_IW_SLC__1SDV_20181008T052755_20181008T052822_024040_02A081_3524_deb_Orb_ML_msk.nc'
img         = nc.Dataset(img_fp)['Intensity_VV_db']
#img_avg3x3     = moving_mean(img, 3)

#x      = []
#y      = []
#value = []
#avg3x3    = []
std3x3    = []

ti = time()
k=1
for yi in range(    2, img.shape[0]-2, 3):
    for xi in range(2, img.shape[1]-2, 3):
#for yi in range(    3000, 4000, 3):
#    for xi in range(1000, 2000, 3):
        #x.append(xi)
        #y.append(yi)
        #value.append(img[yi,xi])
        #avg3x3.append(img_avg3x3[yi,xi])
        std3x3.append(np.std(img[ yi-1:yi+2,
                                  xi-1:xi+2 ]))
        tf      = time()
        dt      = tf-ti
        prog    = 100*k/(img.size/3**2)
        v       = prog/dt
        left   = (100-prog)/v
        #print(left)
        print(f'Lasts {strftime("%Hh %Mmin %Ss", gmtime(left))}.\tIteration = {k}.')
        k+=1


#np.save('all_x',np.array(x));         del x
#np.save('all_y',np.array(y));         del y
#np.save('all_value',np.array(value)); del value
#np.save('all_avg3x3',np.array(avg3x3));  del avg3x3
np.save('all_std3x3', np.array(std3x3));   del std3x3
