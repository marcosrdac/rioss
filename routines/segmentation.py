#!/usr/bin/env python3

import re
import numpy as np
import netCDF4 as nc
from keras.models import load_model  # needs to be imported here
from pickle import loads
import matplotlib.pyplot as plt
from os import listdir, makedirs
from os.path import expanduser, isdir, splitext, join, basename
from functions import get_mwa, get_mrwa, mwsd


MODELS = join('..', 'models')
MODEL_NAME = 'segmentation_val_mwa15_mwsd15_mrwa1625_ARCH_5_1_5'
SEGMENTATION_MODEL = join(MODELS, f'{MODEL_NAME}.h5')
SEGMENTATION_MODEL_SCALER = join(MODELS, f'{MODEL_NAME}_scaler')


segmentation_model = load_model(SEGMENTATION_MODEL)
with open(SEGMENTATION_MODEL_SCALER, 'rb') as f:
    segmentation_model_scaler = loads(f.read())


FEATURES = [
    {
        'name': 'val',
        'function': lambda img: img,
    },
    {
        'name': 'mwa15x15',
        'function': lambda img: get_mwa(7)(img),
    },
    {
        'name': 'mwsd15x15',
        'function': lambda img: mwsd(img, 7),
    },
    {
        'name': 'mrwa1625',
        'function': lambda img: get_mrwa(25, 16)(img),
    },
]


# LABELS = {
#     0: 'oil',
#     1: 'sea',
# }


def segmentate(img):

    def calculate_features(img):
        n_functions = len(FEATURES)
        feats = np.empty((img.size, n_functions))
        print('function_name:')
        for i, feature in enumerate(FEATURES):
            print(f"{feature['name']}")
            feats[:, i] = feature['function'](img).flatten()
        return(feats)

    # print('segmentation')
    X = calculate_features(img)
    X = np.where(np.isnan(X), 0, X)  # check if it's okay
    X = segmentation_model_scaler.transform(X)
    # print('actually predicting')
    y = np.argmax(segmentation_model.predict(X), 1)
    return(y.reshape(img.shape))


#print('\n'*80)
#print('starting')
#ncd = nc.Dataset(
#    '/home/marcosrdac/tmp/los/sar/' +
#    'subset_0_of_S1A_IW_SLC__1SDV_20181009T171427_20181009T171454_024062_02A131_CCBB_Cal_Orb_Deb_ML_dB.nc')
## 'subset_0_of_S1B_IW_SLC__1SDV_20190319T181151_20190319T181219_015427_01CE45_2311.nc')
#ncvar = ncd.variables['Sigma0_VV_db']
## var = np.array(ncvar)[:2000, 2000:3000]
## var = np.array(ncvar)
#var = np.array(ncvar)[:30, :30]
#
## var = np.load('/mnt/hdd/tmp/sar/test.npy')
##
#fig, axes = plt.subplots(1, 2, dpi=300, figsize=(10, 4))
#axes[0].imshow(var)
#axes[1].imshow(segmentate(var))
#fig.tight_layout()
#fig.savefig('segmodel_5_2_1_5_10.png')
#plt.show()
