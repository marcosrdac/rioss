#!/usr/bin/env python


# DEPENDENCIES
import re
from os import listdir
from os.path import splitext, join, basename
import numpy as np
import netCDF4 as nc
from PIL import Image
# DEFINED
from parameters import CLASSIFICATION_INPUT_DATA, CLASSIFICATION_INPUT_MASKS, \
    CLASSIFICATION_BLOCKS_OUTPUT, CLASSIFICATION_FEATURES_OUTPUT
from routines.functions import listifext, first_channel, sample_min_dist
# temporary
import matplotlib.pyplot as plt


BLOCK_WINDOW_SIDE = 512


FUNCTIONS = {
    'val': (lambda img: img),
}

LABEL_NUMBERS = {
    'oil': 0,
    'look_alike': 1,
    'sea': 2,
    'city': 3,
}

# preconfiguration
input_data_fps = listifext(CLASSIFICATION_INPUT_DATA, 'nc', fullpath=True)
input_masks_fps = listifext(CLASSIFICATION_INPUT_MASKS, 'jpg', fullpath=True)

for ncf in input_data_fps:
    print(f'Using file: "{ncf}".')
    print(f'  Mask files found:')
    # getting mask data
    base = splitext(basename(ncf))[0]
    mask_files = [f for f in input_masks_fps if base in f]
    for mask_fp in mask_files:
        mask_name = re.findall(r'.*_([^_]*).jpg', mask_fp)[0]
        print(f'    - {mask_name} ({mask_fp})')
        mask = first_channel(np.array(Image.open(mask_fp)))
        sampled_mask_coords = sample_min_dist(mask, BLOCK_WINDOW_SIDE)
        print(sampled_mask_coords)
#        mask_npoints = np.sum(mask)
#        oil_npoints, sea_npoints, city_npoints = 3 * [0]
#        has_oil, has_sea, has_city = 3 * [False]
#        if mask_name == 'oil':
#            has_oil = True
#            oil_mask = mask
#            oil_npoints = mask_npoints
#        elif mask_name == 'sea':
#            has_sea = True
#            sea_mask = mask
#            sea_npoints = mask_npoints
#        elif mask_name == 'city':
#            has_city = True
#            city_mask = mask
#            city_npoints = mask_npoints
#    if not any([has_oil, has_sea, has_city]):  # if no jpg mask found
#        print(
#            f'    None\n'
#            f'Please add its masks to "{CLASSIFICATION_INPUT_MASKS}".\n'
#            f'Skipping it.')
#        continue
#
#    # opening ooriginal data
#    ncd = nc.Dataset(ncf)
#    img = ncd['Sigma0_VV_db']  # (Sentinel-1 band name after db conversion)
#    h, w = img.shape
#
#    total_points = oil_npoints + sea_npoints + city_npoints
#    labels = np.empty(total_points)
#    labels[:oil_npoints] = LABEL_NUMBERS['oil']
#    labels[oil_npoints:oil_npoints+sea_npoints] = LABEL_NUMBERS['sea']
#    labels[oil_npoints+sea_npoints:total_points] = LABEL_NUMBERS['city']
#    np.save(join(CLASSIFICATION_FEATURES_OUTPUT,
#                 f'{base}_{mask_name}.npy'), labels)
#    del(labels)
#
#    for function_name, function in FUNCTIONS.items():
#        resultimg = function(img)
#        if has_oil:
#            oil_points = resultimg[oil_mask]
#        if has_sea:
#            sea_points = resultimg[sea_mask]
#        if has_city:
#            city_points = resultimg[city_mask]
#        all_masked = np.concatenate([oil_points, sea_points, city_points])
#        np.save(join(CLASSIFICATION_FEATURES_OUTPUT,
#                     f'{base}_{function_name}.npy'), all_masked)
#        del(all_masked)
#        del(oil_points)
#        del(city_points)
