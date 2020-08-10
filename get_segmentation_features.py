#!/usr/bin/env python


# NECESSARY TODO
#   update FUNCTIONS to get_mwa(r) description


# DEPENDENCIES
import re
from os import listdir
from os.path import splitext, join, basename
import numpy as np
import netCDF4 as nc
from PIL import Image
# DEFINED
from parameters import SEGMENTATION_INPUT_DATA, SEGMENTATION_INPUT_MASKS, \
    SEGMENTATION_FEATURES_OUTPUT
from routines.functions import mwa, mcwa, mwsd


FUNCTIONS = {
    'val':       (lambda img: img),
    'mwa3x3':    (lambda img: get_mwa(1)(img)),
    'mwsd3x3':   (lambda img: mwsd(img, 1)),
    'mwa15x15':  (lambda img: get_mwa(7)(img)),
    'mwsd15x15': (lambda img: mwsd(img, 7)),
    'mcwa16_25': (lambda img: get_mcwa(16, 25)(img)),
}

LABEL_NUMBERS = {
    'oil': 0,
    'sea': 1,
    # 'terrain': 2,  # not used
}


# preconfiguration
input_data_fps = listdir(SEGMENTATION_INPUT_DATA)
input_masks_fps = listdir(SEGMENTATION_INPUT_MASKS)
input_data_fps = \
    [f for f in input_data_fps if splitext(f)[-1] == '.nc']
input_masks_fps = \
    [f for f in input_masks_fps if splitext(f)[-1] == '.jpg']
input_data_fps = [join(SEGMENTATION_INPUT_DATA, f)
                  for f in input_data_fps]
input_masks_fps = [join(SEGMENTATION_INPUT_MASKS, f)
                   for f in input_masks_fps]


for ncf in input_data_fps:
    print(f'Using file: "{ncf}".')
    print(f'  Mask files found:')
    # getting mask data
    base = splitext(basename(ncf))[0]
    mask_files = [f for f in input_masks_fps if base in f]
    for mask_fp in mask_files:
        mask_name = re.findall(r'.*_([^_]*).jpg', mask_fp)[0]
        print(f'    - {mask_name} ({mask_fp})')
        # why is it still three-channel?
        mask = np.array(Image.open(mask_fp))[:, :, 0]
        mask = np.where(mask > 128, True, False)
        mask_npoints = np.sum(mask)
        oil_npoints = sea_npoints = terrain_npoints = 0
        has_oil = has_sea = has_terrain = False
        if mask_name == 'oil':
            has_oil = True
            oil_mask = mask
            oil_npoints = mask_npoints
        elif mask_name == 'sea':
            has_sea = True
            sea_mask = mask
            sea_npoints = mask_npoints
        elif mask_name == 'terrain':
            has_terrain = True
            terrain_mask = mask
            terrain_npoints = mask_npoints
    if not any([has_oil, has_sea, has_terrain]):  # if no jpg mask found
        print(
            f'    None\n'
            f'Please add its masks to "{SEGMENTATION_INPUT_MASKS}".\n'
            f'Skipping it.')
        continue

    # opening ooriginal data
    ncd = nc.Dataset(ncf)
    img = ncd['Sigma0_VV_db']  # (Sentinel-1 band name after db conversion)
    h, w = img.shape

    total_points = oil_npoints + sea_npoints + terrain_npoints
    labels = np.empty(total_points)
    labels[:oil_npoints] = LABEL_NUMBERS['oil']
    labels[oil_npoints:oil_npoints+sea_npoints] = LABEL_NUMBERS['sea']
    labels[oil_npoints+sea_npoints:total_points] = LABEL_NUMBERS['terrain']
    np.save(join(SEGMENTATION_FEATURES_OUTPUT,
                 f'{base}_{mask_name}.npy'), labels)
    del(labels)

    for function_name, function in FUNCTIONS.items():
        resultimg = function(img)
        if has_oil:
            oil_points = resultimg[oil_mask]
        if has_sea:
            sea_points = resultimg[sea_mask]
        if has_terrain:
            terrain_points = resultimg[terrain_mask]
        all_masked = np.concatenate([oil_points, sea_points, terrain_points])
        np.save(join(SEGMENTATION_FEATURES_OUTPUT,
                     f'{base}_{function_name}.npy'), all_masked)
        del(all_masked)
        del(oil_points)
        del(terrain_points)
