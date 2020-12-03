#!/usr/bin/env python3

from os import listdir, makedirs
from os.path import expanduser, isdir, splitext, join, basename
import re
import numpy as np
from scipy.stats import skew, kurtosis
from pickle import loads
from routines.functions import discarray
from routines.functions import lacunarity, perimeter, area, get_glcm, grad
# from segmentation import segmentate
from thresh_segmentation import segmentate
from adth_block_functions import BLOCK_FUNCTIONS
from parameters import CLASSIFICATION_CATEGORIES


MODELS = 'models'
MODEL_NAME = 'adth_random_forests_classification_model.bin'
# CLASSIFICATION_MODEL_SCALER = join(MODELS, f'{MODEL_NAME}_scaler')
FEATURE_NAMES_NAME = 'adth_random_forests_classification_features.bin'
CLASSIFICATION_MODEL = join(MODELS, MODEL_NAME)
CLASSIFICATION_FEATURE_NAMES = join(MODELS, FEATURE_NAMES_NAME)

with open(CLASSIFICATION_MODEL, 'rb') as f:
    classification_model = loads(f.read())
    classification_model.n_jobs = -1

# with open(CLASSIFICATION_FEATURE_NAMES, 'rb') as f:
    # FEATURE_NAMES = loads(f.read())
    # # print(FEATURE_NAMES)

FEATURE_NAMES = ['std', 'skew', 'kurt', 'entropy', 'fgperoarea', 'psdfd',
                 'bcfd', 'bclac', 'gradmax', 'gradmean', 'gradmeanomedian']

FEATURES = [{'name': name, 'function': BLOCK_FUNCTIONS[name]}
                 for name in FEATURE_NAMES]

# print(FEATURES)

def classify(img, proba=False):
    def calculate_features(img):
        n_functions = len(FEATURES)
        feats = np.empty(n_functions)
        segmented = segmentate(img)
        glcm = None
        segglcm = None
        _grad = grad(img)

        for i, feature in enumerate(FEATURES):
            print(f"    Calculating: {feature['name']}")
            feats[i] = feature['function'](img, segmented, glcm, segglcm, _grad)
            # print(feats[i])
        return(feats)

    # return calculate_features(img)
    # Add verbose option
    print(f"    Calculating block features.")
    x = calculate_features(img)[None, :]
    print(f"    Dealing with NaNs.")
    x = np.where(np.isnan(x), -1, x)  # check if it's okay
    x = np.where(~np.isfinite(x), -1, x)  # check if it's okay
    # print(f"    Scaling data.")
    # X = segmentation_model_scaler.transform(X)
    print(f"    Perfforming segmentation of block.")
    if proba:
        y = classification_model.predict_proba(x)[0]
    else:
        y = classification_model.predict(x)
    return y


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
