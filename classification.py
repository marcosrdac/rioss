#!/usr/bin/env python3

from os import listdir, makedirs
from os.path import expanduser, isdir, splitext, join, basename
import re
import numpy as np
from scipy.stats import skew, kurtosis
from pickle import loads
from routines.functions import discarray
from routines.functions import grad, lacunarity, perimeter, area, get_glcm
from segmentation import segmentate
from block_functions import BLOCK_FUNCTIONS
from parameters import MODELS, CLASSIFICATION_CATEGORIES


DETECTOR_MODEL = join(MODELS, 'rf_2.bin')
CLASSIFIER_MODEL = join(MODELS, 'rf_7.bin')


# model files are pickled as [feature_names, model]

try:
    with open(DETECTOR_MODEL, 'rb') as f:
        detector_feature_names, detector = loads(f.read())
        detector.n_jobs = -1
        detector_features = [{'name': name,
                                'function': BLOCK_FUNCTIONS[name]}
                                for name in detector_feature_names]
except FileNotFoundError:
    pass
except TypeError:
    pass

try:
    with open(CLASSIFIER_MODEL, 'rb') as f:
        classifier_feature_names, classifier = loads(f.read())
        classifier.n_jobs = -1
        classifier_features = [{'name': name,
                                'function': BLOCK_FUNCTIONS[name]}
                               for name in classifier_feature_names]
except FileNotFoundError:
    pass
except TypeError:
    pass


def calculate_features(img, features, verbose=True):
    _print = print if verbose else lambda *x, **y: None
    feats = np.empty(len(features))
    segmented = segmentate(img)
    glcm = get_glcm(img)
    segglcm = get_glcm(segmented)
    _grad = grad(img)

    _print(f"    Calculating:", end=' ')
    for i, feature in enumerate(features):
        end = ', ' if i < len(features)-1 else '.\n'
        _print(f"{feature['name']}", end=end)
        feats[i] = feature['function'](img, segmented, glcm, segglcm, _grad)
    return feats

def apply_generic_model(img, model, features, scaler=None, proba=False, verbose=True):
    _print = print if verbose else lambda *x, **y: None
    _print(f"    Calculating block features.")
    x = calculate_features(img, features, verbose=verbose)[None, :]
    _print(f"    Dealing with NaNs.")
    x = np.where(np.isnan(x), 0, x)      # to be made sure
    x = np.where(~np.isfinite(x), 0, x)  # to be made sure
    if scaler:
        _print(f"    Scaling data.")
        X = scaler.transform(x)
    _print(f"    Applying model to block.", end='\n\n')
    if proba:
        y = model.predict_proba(x)[0]
    else:
        y = model.predict(x)
    return y

def calculate_detector_features(img, verbose=True):
    return calculate_features(img, detector_features, verbose)

def calculate_classifier_features(img, verbose=True):
    return calculate_features(img, classifier_features, verbose)

def apply_detector(img, proba=False, verbose=True):
    return apply_generic_model(img, detector, detector_features, proba=proba, verbose=verbose)

def apply_classifier(img, proba=False, verbose=True):
    return apply_generic_model(img, classifier, classifier_features, proba=proba, verbose=verbose)


if __name__ == '__main__':
    import netCDF4 as nc
    import matplotlib.pyplot as plt
    from random import shuffle
    folder = '/mnt/hdd/home/tmp/los/data/classification_blocks'
    files = [join(folder, f) for f in listdir(folder)]
    shuffle(files)
    for f in files:
        img = discarray(f)
        c = apply_classifier(img)
        # c =  apply_detector(img)

        plt.suptitle(basename(f))
        # plt.title(f'{CLASSIFICATION_CATEGORIES[int(c)]["name"]}')
        plt.title(f'{c}')
        plt.imshow(img)
        plt.show()
