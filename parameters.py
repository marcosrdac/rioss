#!/usr/bin/env python

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


# CLASSIFICATION MODEL

CLASSIFICATION_INPUT_DATA = ORIGINAL_DATA
CLASSIFICATION_INPUT_MASKS = join(BASE, 'classification_masks')
CLASSIFICATION_BLOCKS_OUTPUT = join(BASE, 'classification_blocks')
CLASSIFICATION_FEATURES_OUTPUT = join(BASE, 'classification_model_features')
