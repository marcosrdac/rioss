#!/usr/bin/env python3

import re
import numpy as np
import netCDF4 as nc
# from keras.models import load_model  # needs to be imported here
from pickle import loads
import matplotlib.pyplot as plt
from os import listdir, makedirs
from os.path import expanduser, isdir, splitext, join, basename
# from .functions import get_mwa, get_mrwa, mwsd
from functions import get_mwa, get_mrwa, mwsd

if __name__ == '__main__':
    ROOT = '..'
else:
    ROOT = '.'

MODELS = join(ROOT, 'models')
MODEL_NAME = 'segmentation_gmm_mwa15_LABELS_1_0.bin'
SEGMENTATION_MODEL = join(MODELS, MODEL_NAME)
# SEGMENTATION_MODEL_SCALER = join(MODELS, f'{MODEL_NAME}_scaler')

with open(SEGMENTATION_MODEL, 'rb') as f:
    segmentation_model = loads(f.read())


FEATURES = [
    # {
        # 'name': 'val',
        # 'function': lambda img: img,
    # },
    {
        'name': 'mwa15x15',
        'function': lambda img: get_mwa(7)(img),
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
        for i, feature in enumerate(FEATURES):
            print(f"    Calculating: {feature['name']}")
            feats[:, i] = feature['function'](img).flatten()
        return(feats)

    # Add verbose option

    print(f"    Calculating block features.")
    X = calculate_features(img)
    print(f"    Dealing with NaNs.")
    X = np.where(np.isnan(X), 0, X)  # check if it's okay
    # print(f"    Scaling data.")
    # X = segmentation_model_scaler.transform(X)
    print(f"    Perfforming segmentation of block.")
    y = segmentation_model.predict(X)
    return(y.reshape(img.shape))


if __name__ == '__main__':
    ncd = nc.Dataset(
        '/home/marcosrdac/tmp/los/sar/' +
        'subset_0_of_S1A_IW_SLC__1SDV_20181009T171427_20181009T171454_024062_02A131_CCBB_Cal_Orb_Deb_ML_dB.nc')
    ## 'subset_0_of_S1B_IW_SLC__1SDV_20190319T181151_20190319T181219_015427_01CE45_2311.nc')
    ncvar = ncd.variables['Sigma0_VV_db']
    var = np.array(ncvar)

    import seaborn as sns
    sns.distplot(get_mwa(7)(var).flatten())
    plt.show()


#    segmented = segmentate(var)
#
#    fig, axes = plt.subplots(1, 2, dpi=300, figsize=(10, 4))
#    axes[0].imshow(var)
#    axes[1].imshow(segmented)
#    fig.tight_layout()
#    fig.savefig('segmodel_5_2_1_5_10.png')
#    plt.show()
#
