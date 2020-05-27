#!/usr/bin/env python3

import numpy as np
import netCDF4 as nc
import scipy as scp
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from time import time, strftime, gmtime
from os import listdir, makedirs
from os.path import expanduser, isdir, splitext, join
from routines.functions import mwsd, mwa, mcwa


# user defined
INPUT_DATA_DIR =  expanduser('~/data/0_original/')
INPUT_MASKS_DIR = expanduser('~/data/2_masks/')
OUTPUT_DIR =      expanduser('~/data/3_segmodel_creation')

# preconfiguration
input_data_dir_fps = listdir(INPUT_DATA_DIR)
input_masks_dir_fps = listdir(INPUT_MASKS_DIR)
input_data_dir_fps = [f for f in input_data_dir_fps if splitext(f)[-1]=='.nc']
input_data_dir_fps =  [join(INPUT_DATA_DIR, f) for f in input_data_dir_fps]
input_masks_dir_fps = [join(INPUT_MASKS_DIR, f) for f in input_masks_dir_fps]


for ncf in input_data_dir_fps:
    # SENTINEL-1 CONFIGURATION
    print(f'Using file: "{ncf}".')
    ncd = nc.Dataset(ncf)
    img = ncd['Sigma0_VV_db']
    img_mwa3x3 = mwa(img, 1)
    img_mwa15x15 = mwa(img, 7)
#    #img_mwa5x5     = mwa(img, 1)
#    #img_mcwa2516 = mcwa(img, 25, 16)
#    bg_sea_msk  = np.load('../data/bg_sea_msk.npy')
#    fg_spot_msk = np.load('../data/fg_spot_msk.npy')
#
#total_labeled = np.sum(bg_sea_msk) + np.sum(fg_spot_msk)
#
#shape = img.shape
#
#x             = []
#y             = []
#value         = []
#avg3x3        = []
#circavg125i16 = []
#std3x3        = []
#label         = [] # 0 bg_sea, 1 fg_spot
#
#k=0
#add = 'test'+'_'
#for yi in range(    3000, 4000, 3):
#    for xi in range(1000, 2000, 3):
##for yi in range(    2, img.shape[0]-2, 3):
##    for xi in range(2, img.shape[1]-2, 3):
#        labeled = 0
#        if bg_sea_msk[yi,xi]:
#            labeli  = 0
#            labeled = 1
#        elif fg_spot_msk[yi,xi]:
#            labeli  = 1
#            labeled = 1
#        if labeled == 1:
#            x.append(xi)
#            y.append(yi)
#            value.append(img[yi,xi])
#            avg3x3.append(img_avg3x3[yi,xi])
#            circavg125i16.append(img_circavg125i16[yi,xi])
#            std3x3.append(np.std(img[ yi-1:yi+2,
#                                      xi-1:xi+2 ]))
#            label.append(labeli)
#            tf      = time()
#            dt      = tf-ti
#            prog    = 100*k/(img.size/3**2)
#            v       = prog/dt
#            left   = (100-prog)/v
#            #print(left)
#            print(f'Lasts {strftime("%Hh %Mmin %Ss", gmtime(left))}.\tIteration = {k}.')
#            k+=1
#
#
#np.save(add+'x',np.array(x));             del x
#np.save(add+'y',np.array(y));             del y
#np.save(add+'value',np.array(value));     del value
#np.save(add+'avg3x3',np.array(avg3x3));   del avg3x3
#np.save(add+'std3x3',np.array(std3x3));   del std3x3
#np.save(add+'label',np.array(label));     del label
