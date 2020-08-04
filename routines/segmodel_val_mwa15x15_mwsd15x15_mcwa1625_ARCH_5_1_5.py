#!/usr/bin/env python3

import re
import numpy as np
import netCDF4 as nc
from keras.models import load_model  # needs to be imported here
from pickle import loads
import scipy as scp
import matplotlib.pyplot as plt
import seaborn as sns
from os import listdir, makedirs
from os.path import expanduser, isdir, splitext, join, basename
from functions import get_mwa, get_mrwa, mwsd
segmodel = load_model('segmodel_val_mwa15x15_mwsd15x15_mcwa1625_ARCH_5_1_5.h5')
with open('segmodel_val_mwa15x15_mwsd15x15_mcwa1625_ARCH_5_1_5_scaler', 'rb') as f:
    scaler = loads(f.read())
#
# user defined
INPUT_DATA_DIR = expanduser('~/data/0_original/')

mwa3x3 = get_mwa(1)
mwa15x15 = get_mwa(7)
mcwa1625 = get_mrwa(25, 16)

FUNCTIONS = {
    'val': (lambda img: img),
    'mwa15x15': (lambda img: mwa15x15(img)),
    'mwsd15x15': (lambda img: mwsd(img, 7)),
    'mrwa1625': (lambda img: mcwa1625(img)),
}

LABELS = {
    0: 'oil',
    1: 'sea',
}


def calculate_feats(img, functions):
    n_functions = len(functions)
    feats = np.empty((img.size, n_functions))
    print('function_name')
    for i, (function_name, function) in enumerate(functions.items()):
        print(f'{function_name}')
        feats[:, i] = function(img).flatten()
    return(feats)


def segmentate(img, functions=FUNCTIONS):
    print('segmentation')
    X = calculate_feats(img, functions)
    X = np.where(np.isnan(X), 0, X)  # check if it's okay
    X = scaler.transform(X)
    print('actually predicting')
    y = np.argmax(segmodel.predict(X), 1)
    return(y.reshape(img.shape))


print('\n'*80)
print('starting')
ncd = nc.Dataset(
        '/home/marcosrdac/tmp/los/sar/' +
        'subset_0_of_S1A_IW_SLC__1SDV_20181009T171427_20181009T171454_024062_02A131_CCBB_Cal_Orb_Deb_ML_dB.nc')
        #'subset_0_of_S1B_IW_SLC__1SDV_20190319T181151_20190319T181219_015427_01CE45_2311.nc')
ncvar = ncd.variables['Sigma0_VV_db']
#var = np.array(ncvar)[:2000, 2000:3000]
var = np.array(ncvar)
#var = np.array(ncvar)[:30,:30]

#var = np.load('/mnt/hdd/tmp/sar/test.npy')
#
fig, axes = plt.subplots(1, 2, dpi=300, figsize=(10, 4))
axes[0].imshow(var)
axes[1].imshow(segmentate(var))
fig.tight_layout()
fig.savefig('segmodel_5_2_1_5_10.png')
plt.show()
