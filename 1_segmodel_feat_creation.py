#!/usr/bin/env python3

import numpy as np
import re
import netCDF4 as nc
import scipy as scp
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from time import time, strftime, gmtime
from os import listdir, makedirs
from os.path import expanduser, isdir, splitext, join, basename
from routines.functions import mwa, mcwa, mwsd


# user defined
INPUT_DATA_DIR = expanduser('~/data/0_original/')
INPUT_MASKS_DIR = expanduser('~/data/2_masks/')
OUTPUT_DIR = expanduser('~/data/3_segmodel_creation')


FUNCTIONS = {
    'val': (lambda img: img),
    'mwa3x3': (lambda img: mwa(img, 1)),
    'mwsd3x3': (lambda img: mwsd(img, 1)),
    'mwa15x15': (lambda img: mwa(img, 7)),
    'mwsd15x15': (lambda img: mwsd(img, 7)),
    'mcwa16_25': (lambda img: mcwa(img, 25, 16)),
}

LABEL_NUMBERS = {
    'oil': 0,
    'sea': 1,
    'city': 2,
}


# preconfiguration
input_data_dir_fps = listdir(INPUT_DATA_DIR)
input_masks_dir_fps = listdir(INPUT_MASKS_DIR)
input_data_dir_fps = \
    [f for f in input_data_dir_fps if splitext(f)[-1] == '.nc']
input_masks_dir_fps = \
    [f for f in input_masks_dir_fps if splitext(f)[-1] == '.jpg']
input_data_dir_fps = [join(INPUT_DATA_DIR, f) for f in input_data_dir_fps]
input_masks_dir_fps = [join(INPUT_MASKS_DIR, f) for f in input_masks_dir_fps]


for ncf in input_data_dir_fps:
    print(f'Using file: "{ncf}".')
    print(f'  Mask files found:')
    # getting mask data
    base = splitext(basename(ncf))[0]
    mask_files = [f for f in input_masks_dir_fps if base in f]
    for mask_fp in mask_files:
        mask_name = re.findall(r'.*_([^_]*).jpg', mask_fp)[0]
        print(f'    - {mask_name} ({mask_fp})')
        # why is it still three-channel?
        mask = np.array(Image.open(mask_fp))[:, :, 0]
        mask = np.where(mask > 128, True, False)
        mask_npoints = np.sum(mask)
        oil_npoints, sea_npoints, city_npoints = 3 * [0]
        has_oil, has_sea, has_city = 3 * [False]
        if mask_name == 'oil':
            has_oil = True
            oil_mask = mask
            oil_npoints = mask_npoints
        elif mask_name == 'sea':
            has_sea = True
            sea_mask = mask
            sea_npoints = mask_npoints
        elif mask_name == 'city':
            has_city = True
            city_mask = mask
            city_npoints = mask_npoints
    if not any([has_oil, has_sea, has_city]):  # if no jpg mask found
        print(
            f'    None\n'
            f'Please add its masks to "{INPUT_MASKS_DIR}".\n'
            f'Skipping it.')
        continue

    # opening ooriginal data
    ncd = nc.Dataset(ncf)
    img = ncd['Sigma0_VV_db']  # (Sentinel-1 band name after db conversion)
    h, w = img.shape

    total_points = oil_npoints + sea_npoints + city_npoints
    labels = np.empty(total_points)
    labels[:oil_npoints] = LABEL_NUMBERS['oil']
    labels[oil_npoints:oil_npoints+sea_npoints] = LABEL_NUMBERS['sea']
    labels[oil_npoints+sea_npoints:total_points] = LABEL_NUMBERS['city']
    np.save(join(OUTPUT_DIR, f'{base}_{mask_name}.npy'), labels)
    del(labels)

    for function_name, function in FUNCTIONS.items():
        resultimg = function(img)
        if has_oil:
            oil_points = resultimg[oil_mask]
        if has_sea:
            sea_points = resultimg[sea_mask]
        if has_city:
            city_points = resultimg[city_mask]
        all_masked = np.concatenate([oil_points, sea_points, city_points])
        np.save(join(OUTPUT_DIR, f'{base}_{function_name}.npy'), all_masked)
        del(all_masked)
        del(oil_points)
        del(city_points)
