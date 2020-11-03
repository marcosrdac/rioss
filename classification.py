#!/usr/bin/env python3

import re
import numpy as np
from scipy.stats import skew, kurtosis
import netCDF4 as nc
# from keras.models import load_model  # needs to be imported here
from pickle import loads
import matplotlib.pyplot as plt
from os import listdir, makedirs
from os.path import expanduser, isdir, splitext, join, basename
# from .functions import get_mwa, get_mrwa, mwsd
from routines.functions import discarray, lacunarity, perimeter, area, get_glcm, get_homogeneity, get_dissimilarity, get_correlation
import routines.psd as psd
from segmentation import segmentate

if __name__ == '__main__':
    ROOT = '.'
else:
    ROOT = '.'

MODELS = join(ROOT, 'models')
MODEL_NAME = '../models/random_forests_classification_model.bin'
# SEGMENTATION_MODEL_SCALER = join(MODELS, f'{MODEL_NAME}_scaler')
# FEATURE_NAMES_NAME = '../models/random_forests_classification_features.bin'
CLASSIFICATION_MODEL = join(MODELS, MODEL_NAME)
# CLASSIFICATION_FEATURE_NAMES = join(MODELS, FEATURE_NAMES_NAME)

with open(CLASSIFICATION_MODEL, 'rb') as f:
    classification_model = loads(f.read())

print(classification_model.classes_)

# with open(CLASSIFICATION_FEATURE_NAMES, 'rb') as f:
# feature_names = loads(f.read()))
feature_names = ['std', 'psdfd', 'skew', 'fgrelarea', 'bclac', 'kurt',
                 'fgperoarea', 'bgmean', 'segglcmcorr', 'fgstd', 'glcmdiss',
                 'segglcmhomo']

# segmented here is true when pixel is NOT foreground (it may not be the case
# for other segmentation functions)  ALTER BELOW TO MATCH IT
FEATURES_DICT = {
    'std': lambda img, segmented, glcm, segglcm: np.std(img),
    'psdfd': lambda img, segmented, glcm, segglcm: psd.fractal_dimension(img),
    'skew': lambda img, segmented, glcm, segglcm: skew(img.flatten()),
    'fgrelarea': lambda img, segmented, glcm, segglcm: area(segmented)/np.multiply(*img.shape),
    'bclac': lambda img, segmented, glcm, segglcm: lacunarity(img),
    'kurt': lambda img, segmented, glcm, segglcm: kurtosis(img.flatten()),
    'fgperoarea': lambda img, segmented, glcm, segglcm: perimeter(segmented)/area(segmented),
    'bgmean': lambda img, segmented, glcm, segglcm: np.mean(img[~segmented]),
    'segglcmcorr': lambda img, segmented, glcm, segglcm: get_correlation(segglcm),
    'fgstd': lambda img, segmented, glcm, segglcm: np.std(img[segmented]),
    'glcmdiss': lambda img, segmented, glcm, segglcm: get_dissimilarity(glcm),
    'segglcmhomo': lambda img, segmented, glcm, segglcm: get_homogeneity(segglcm),
}

FEATURES = [{'name': n, 'function': f} for n, f in FEATURES_DICT.items()]

# LABELS = {
# 0: 'oil',
# 1: 'sea',
# }


def classificate(img):
    def calculate_features(img):
        n_functions = len(FEATURES)
        feats = np.empty(n_functions)
        segmented = segmentate(img)
        glcm = get_glcm(img)
        segglcm = get_glcm(segmented)

        for i, feature in enumerate(FEATURES):
            print(f"    Calculating: {feature['name']}")
            feats[i] = feature['function'](img, segmented, glcm, segglcm)
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
    from random import shuffle
    folder = '/mnt/hdd/home/tmp/los/data/classification_blocks'
    files = [join(folder, f) for f in listdir(folder)]
    shuffle(files)
    for f in files:
        img = discarray(f)
        c = classificate(img)

        plt.title(f'{c}')
        plt.imshow(img)
        plt.show()

        # fig, axes = plt.subplots(1, 2, dpi=300, figsize=(10, 4))
        # axes[0].imshow(var)
        # axes[1].imshow(segmented)
        # fig.tight_layout()
        # fig.savefig('segmodel_5_2_1_5_10.png')
        # plt.show()
