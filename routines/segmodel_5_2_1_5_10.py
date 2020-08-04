#!/usr/bin/env python3

import re
import numpy as np
import netCDF4 as nc
from keras.models import load_model  # needs to be imported here
import scipy as scp
import matplotlib.pyplot as plt
import seaborn as sns
from os import listdir, makedirs
from os.path import expanduser, isdir, splitext, join, basename
from functions import get_mwa, get_mrwa, mwsd
segmodel = load_model('segmodel_5_2_1_5_10.h5')
#
# user defined
INPUT_DATA_DIR = expanduser('~/data/0_original/')

mwa3x3 = get_mwa(1)
mwa15x15 = get_mwa(7)
mcwa1625 = get_mrwa(25, 16)

FUNCTIONS = {
    'mrwa1625': (lambda img: mcwa1625(img)),
    'mwa15x15': (lambda img: mwa15x15(img)),
    'mwsd3x3/mwsd15x15': (lambda img: mwsd(img, 1)/mwsd(img, 7)),
    #'mwsd3x3/mwsd15x15': (lambda img: mwa3x3(img)/mwa15x15(img)),
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
    print('actually predicting')
    y = np.argmax(segmodel.predict(X), 1)
    return(y.reshape(img.shape))


print('\n'*80)
print('starting')
ncd = nc.Dataset(
        '/home/marcosrdac/tmp/los/sar/' +
        'subset_0_of_S1B_IW_SLC__1SDV_20190319T181151_20190319T181219_015427_01CE45_2311.nc')
ncvar = ncd.variables['Sigma0_VV_db']
#var = np.array(ncvar)[:2000, 2000:3000]
var = np.array(ncvar)

#var = np.load('/mnt/hdd/tmp/sar/test.npy')
#
fig, axes = plt.subplots(1, 2, dpi=300, figsize=(10, 4))
axes[0].imshow(var)
axes[1].imshow(segmentate(var))
fig.tight_layout()
fig.savefig('segmodel_5_2_1_5_10.png')
plt.show()
