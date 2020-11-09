#!/usr/bin/env python3

from os import listdir, makedirs
from os.path import expanduser, isdir, splitext, join, basename
import re
import numpy as np
from scipy.stats import skew, kurtosis
from pickle import loads
from routines.functions import discarray
from routines.functions import lacunarity, perimeter, area, get_glcm
# from segmentation import segmentate
from thresh_segmentation import segmentate
from block_functions import BLOCK_FUNCTIONS
from parameters import CLASSIFICATION_CATEGORIES


MODELS = 'models'
MODEL_NAME = 'random_forests_classification_model.bin'
# CLASSIFICATION_MODEL_SCALER = join(MODELS, f'{MODEL_NAME}_scaler')
FEATURE_NAMES_NAME = 'random_forests_classification_features.bin'
CLASSIFICATION_MODEL = join(MODELS, MODEL_NAME)
CLASSIFICATION_FEATURE_NAMES = join(MODELS, FEATURE_NAMES_NAME)

with open(CLASSIFICATION_MODEL, 'rb') as f:
    classification_model = loads(f.read())
    classification_model.n_jobs = -1

with open(CLASSIFICATION_FEATURE_NAMES, 'rb') as f:
    FEATURE_NAMES = loads(f.read())
    # print(FEATURE_NAMES)

FEATURE_NAMES = ['std', 'skew', 'kurt', 'fgmean', 'bgmean', 'fgstd',
                 'fgrelarea', 'fgrelper', 'fgperoarea', 'psdfd', 'bcfd',
                 'bclac', 'segglcmcorr']

FEATURES = [{'name': name, 'function': BLOCK_FUNCTIONS[name]}
                 for name in FEATURE_NAMES]

# print(FEATURES)

def classify(img):
    def calculate_features(img):
        n_functions = len(FEATURES)
        feats = np.empty(n_functions)
        segmented = segmentate(img)
        glcm = get_glcm(img)
        segglcm = get_glcm(segmented)

        for i, feature in enumerate(FEATURES):
            print(f"    Calculating: {feature['name']}")
            feats[i] = feature['function'](img, segmented, glcm, segglcm)
            # print(feats[i])
        return(feats)

    # return calculate_features(img)
    # Add verbose option
    print(f"    Calculating block features.")
    x = calculate_features(img)[None, :]
    print(f"    Dealing with NaNs.")
    x = np.where(np.isnan(x), 0, x)  # check if it's okay
    x = np.where(~np.isfinite(x), 0, x)  # check if it's okay
    # print(f"    Scaling data.")
    # X = segmentation_model_scaler.transform(X)
    print(f"    Perfforming segmentation of block.")
    y = classification_model.predict(x)
    return(y)


if __name__ == '__main__':
    import netCDF4 as nc
    import matplotlib.pyplot as plt
    from random import shuffle
    folder = '/mnt/hdd/home/tmp/los/data/classification_blocks'
    files = [join(folder, f) for f in listdir(folder)]
    shuffle(files)
    for f in files:
        img = discarray(f)
        c = classify(img)

        plt.suptitle(basename(f))
        plt.title(f'{CLASSIFICATION_CATEGORIES[int(c)]["name"]}')
        plt.imshow(img)
        plt.show()
