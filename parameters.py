#!/usr/bin/env python

from routines import psd
from routines import boxcounting
from routines.functions import *
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

WS = 512

CLASSIFICATION_CATEGORIES = [
    {
        'name': 'oil',
        'color': 'red',
    },
    {
        'name': 'biofilm',
        'color': 'forestgreen',
    },
    {
        'name': 'rain',
        'color': 'lightskyblue',
    },
    {
        'name': 'wind',
        'color': 'grey',
    },
    {
        'name': 'sea',
        'color': 'blue',
    },
    {
        'name': 'ship',
        'color': 'darkmagenta',
    },
    {
        'name': 'terrain',
        'color': 'brown',
    },
]


CLASSIFICATION_CATEGORIES_2 = [
    # just used when plotting
    {
        'name': 'oil',
        'color': 'red',
    },
    {
        'name': 'background',
        'color': 'blue',
    },
]


# please don't use spaces in the abbreviation (abbrv)
CLASSIFICATION_FEATURES = [
    {
        'name': 'boxcounting fractal dimension',
        'abbrv': 'bcfd',
        'function': (lambda img, segmented:
                     boxcounting.fractal_dimension(segmented)),
    },
    {
        'name': 'power spectral density fractal dimension',
        'abbrv': 'psdfd',
        'function': (lambda img, segmented:
                     psd.fractal_dimension(img)),
    },
    {
        'name': 'semivariogram fractal dimension',
        'abbrv': 'svfd',
        'function': (lambda img, segmented:
                     semivariogram.fractal_dimension(img, 5/100)),
    },

    {
        'name': 'normalized shannon entropy',
        'abbrv': 'entropy',
        'function': (lambda img, segmented:
                     shannon.norm_entropy(img)),
    },
    {
        'name': 'segmentation mask normalized shannon entropy',
        'abbrv': 'segentropy',
        'function': (lambda img, segmented:
                     shannon.norm_entropy(segmented)),
    },

    {
        'name': 'glcm correlation',
        'abbrv': 'corr',
        'function': (lambda img, segmented:
                     get_correlation(get_glcm(img))),
    },
    {
        'name': 'glcm dissimilarity',
        'abbrv': 'diss',
        'function': (lambda img, segmented:
                     get_dissimilarity(get_glcm(img))),
    },
    {
        'name': 'glcm homogeneity',
        'abbrv': 'homo',
        'function': (lambda img, segmented:
                     get_homogeneity(get_glcm(img))),
    },
    {
        'name': 'glcm energy',
        'abbrv': 'ener',
        'function': (lambda img, segmented:
                     get_energy(get_glcm(img))),
    },
    {
        'name': 'glcm contrast',
        'abbrv': 'cont',
        'function': (lambda img, segmented:
                     get_contrast(get_glcm(img))),
    },

    {
        'name': 'segmented glcm correlation',
        'abbrv': 'segcorr',
        'function': (lambda img, segmented:
                     get_correlation(get_glcm(segmented))),
    },
    {
        'name': 'segmented glcm dissimilarity',
        'abbrv': 'segdiss',
        'function': (lambda img, segmented:
                     get_dissimilarity(get_glcm(segmented))),
    },
    {
        'name': 'segmented glcm homogeneity',
        'abbrv': 'seghomo',
        'function': (lambda img, segmented:
                     get_homogeneity(get_glcm(segmented))),
    },
    {
        'name': 'segmented glcm energy',
        'abbrv': 'segener',
        'function': (lambda img, segmented:
                     get_energy(get_glcm(segmented))),
    },
    {
        'name': 'segmented glcm contrast',
        'abbrv': 'segcont',
        'function': (lambda img, segmented:
                     get_contrast(get_glcm(segmented))),
    },

    {
        'name': 'box counting lacunarity',
        'abbrv': 'bclac',
        'function': (lambda img, segmented:
                     lacunarity(img)),
    },
    {
        'name': 'segmented box counting lacunarity',
        'abbrv': 'segbclac',
        'function': (lambda img, segmented:
                     lacunarity(segmented)),
    },

    {
        'name': 'segmented perimeter',
        'abbrv': 'fgper',
        'function': (lambda img, segmented:
                     perimeter(segmented)),
    },

    {
        'name': 'segmented area',
        'abbrv': 'fgarea',
        'function': (lambda img, segmented:
                     area(segmented)),
    },


    {
        'name': 'spot mean',
        'abbrv': 'fgmean',
        'function': (lambda img, segmented:
                     apply_over_masked(img, segmented, np.mean)),
    },
    {
        'name': 'spot standard deviation',
        'abbrv': 'fgstd',
        'function': (lambda img, segmented:
                     apply_over_masked(img, segmented, np.std)),
    },
    {
        'name': 'spot skewness',
        'abbrv': 'fgskew',
        'function': (lambda img, segmented:
                     apply_over_masked(img.flatten(), segmented.flatten(), skew)),
    },
    {
        'name': 'spot kurtosis',
        'abbrv': 'fgkurt',
        'function': (lambda img, segmented:
                     apply_over_masked(img.flatten(), segmented.flatten(), kurtosis)),
    },


    {
        'name': 'background mean',
        'abbrv': 'bgmean',
        'function': (lambda img, segmented:
                     apply_over_masked(img, ~segmented, np.mean)),
    },
    {
        'name': 'background standard deviation',
        'abbrv': 'bgstd',
        'function': (lambda img, segmented:
                     apply_over_masked(img, ~segmented, np.std)),
    },
    {
        'name': 'background skewness',
        'abbrv': 'bgskew',
        'function': (lambda img, segmented:
                     apply_over_masked(img.flatten(), ~segmented.flatten(), skew)),
    },
    {
        'name': 'background kurtosis',
        'abbrv': 'bgkurt',
        'function': (lambda img, segmented:
                     apply_over_masked(img.flatten(), ~segmented.flatten(), kurtosis)),
    },

    {
        'name': 'mean',
        'abbrv': 'mean',
        'function': (lambda img, segmented:
                     np.mean(img)),
    },
    {
        'name': 'standard deviation',
        'abbrv': 'std',
        'function': (lambda img, segmented:
                     np.std(img)),
    },
    {
        'name': 'skewness',
        'abbrv': 'skew',
        'function': (lambda img, segmented:
                     skew(img.flatten())),
    },
    {
        'name': 'kurtosis',
        'abbrv': 'kurt',
        'function': (lambda img, segmented:
                     kurtosis(img.flatten())),
    },

    {
        'name': 'gradient max',
        'abbrv': 'gradmax',
        'function': (lambda img, segmented:
                     grad_max(img)),
    },

    {
        'name': 'gradient mean',
        'abbrv': 'gradmean',
        'function': (lambda img, segmented:
                     grad_mean(img)),
    },

    {
        'name': 'gradient median',
        'abbrv': 'gradmedian',
        'function': (lambda img, segmented:
                     grad_median(img)),
    },

    # {
        # 'name': 'laplacian max',
        # 'abbrv': 'lapmax',
        # 'function': (lambda img, segmented:
                     # lap_max(img)),
    # },

    {
        'name': 'complexity',
        'abbrv': 'complex',
        'function': (lambda img, segmented:
                     complexity(segmented)),
    },

    {
        'name': 'spreading',
        'abbrv': 'spreading',
        'function': (lambda img, segmented:
                     spreading(segmented)),
    },
]

# print(len(CLASSIFICATION_FEATURES))
