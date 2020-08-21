#!/usr/bin/env python

from routines import psd
from routines import boxcounting
from routines.functions import get_mwa, mwsd, get_mrwa
from routines import shannon
from routines import semivariogram
from os.path import expanduser, join
HOME = expanduser('~')  # get cross platform user folder


# MODIFY ALL FOLDER VARIABLES BELOW IF NECESSARY

# Base folder for all modeling files
# BASE = join(HOME, 'data')               # camobi
BASE = join(HOME, 'tmp', 'los', 'data')  # home


# GENERIC ORIGINAL NETCDF DATA FOLDER
ORIGINAL_DATA = join(BASE, 'original')


# SEGMENTATION MODEL

SEGMENTATION_INPUT_DATA = ORIGINAL_DATA
SEGMENTATION_INPUT_MASKS = join(BASE, 'segmentation_masks')
SEGMENTATION_FEATURES_OUTPUT = join(BASE, 'segmentation_model_features')


SEGMENTATION_FEATURES = {
    'val':       (lambda img: img),
    'mwa3x3':    (lambda img: get_mwa(1)(img)),
    'mwsd3x3':   (lambda img: mwsd(img, 1)),
    'mwa15x15':  (lambda img: get_mwa(7)(img)),
    'mwsd15x15': (lambda img: mwsd(img, 7)),
    'mrwa16_25': (lambda img: get_mrwa(16, 25)(img)),
}


# CLASSIFICATION MODEL

CLASSIFICATION_INPUT_DATA = ORIGINAL_DATA
CLASSIFICATION_INPUT_MASKS = join(BASE, 'classification_masks')
CLASSIFICATION_BLOCKS_OUTPUT = join(BASE, 'classification_blocks')
CLASSIFICATION_FEATURES_OUTPUT = join(BASE, 'classification_model_features')


CLASSIFICATION_CATEGORIES = [
    {
        'name': 'oil',
        'color': 'red',
    },
    {
        'name': 'lookalike',
        'color': 'yellow',
    },
    {
        'name': 'sea',
        'color': 'blue',
    },
    {
        'name': 'terrain',
        'color': 'green',
    },
]

CLASSIFICATION_FEATURES = [
    {
        'name': 'boxcounting fractal dimension',
        'abbrv': 'bcfd',
        'function': (lambda img, segmented:
                     boxcounting.fractal_dimension(segmented)),
    },
    {
        'name': 'power spectrum density fractal dimension',
        'abbrv': 'psdfd',
        'function': (lambda img, segmented:
                     psd.fractal_dimension(img)),
    },
    {
        'name': 'semivariogram fractal dimension',
        'abbrv': 'svfd',
        'function': (lambda img, segmented:
                     semivariogram.fractal_dimension(img)),
    },
    {
        'name': 'shannon entropy',
        'abbrv': 'entropy',
        'function': (lambda img, segmented:
                     shannon.entropy(img)),
    },
]
