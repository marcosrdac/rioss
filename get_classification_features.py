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
from routines.functions import discarray, listifext, first_channel, \
    sample_min_dist
# temporary
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# PRECONFIGURATION

# block side, half-side
WS = 512
HWS = WS//2

FEATURES = [
    {
        'name': 'test',
        'function': (lambda img: 1.),
        'values': None,
    },
    # ADD CREATED FEATURES
]

CATEGORIES = [
    {
        'name': 'oil',
        'color': 'red',
        'has_coords': False,
        'coords': []
    },
    {
        'name': 'lookalike',
        'color': 'yellow',
        'has_coords': False,
        'coords': []
    },
    {
        'name': 'sea',
        'color': 'blue',
        'has_coords': False,
        'coords': []
    },
    {
        'name': 'terrain',
        'color': 'green',
        'has_coords': False,
        'coords': []},
]


input_data_fps = listifext(CLASSIFICATION_INPUT_DATA, 'nc', fullpath=True)
input_masks_fps = listifext(CLASSIFICATION_INPUT_MASKS, 'jpg', fullpath=True)

for ncf in input_data_fps:
    print(f'Using file: "{ncf}".')
    print(f'  Mask files found:')
    # getting mask data
    base = splitext(basename(ncf))[0]
    mask_files = [f for f in input_masks_fps if base in f]

    for c in CATEGORIES:
        c['has_coords'] = False

    for mask_fp in mask_files:
        mask_name = re.findall(r'.*_([^_]*).jpg', mask_fp)[0]
        print(f'    - {mask_name} ({mask_fp})')

        # get first channel of mask if it has more than one
        mask = first_channel(np.array(Image.open(mask_fp)))
        # sample coordinates with min distance equals to a fraction of WS
        sampled_mask_coords = sample_min_dist(mask, int(np.round(4/5*WS)))
        # save results to categories
        if sampled_mask_coords.size:  # if mask is not empty
            for c in CATEGORIES:
                if mask_name == c['name']:
                    c['has_coords'] = True
                    c['coords'] = sampled_mask_coords
                    break
        else:
            print(f'    [ WARNING ] No points in "{mask_name}".')

    # if no coordinates for masks are achieved
    if not any([c['has_coords'] for c in CATEGORIES]):
        print(
            f'    None\n'
            f'Please add its masks to "{CLASSIFICATION_INPUT_MASKS}".\n'
            f'Skipping image.')
        continue

    # opening ooriginal data
    ncd = nc.Dataset(ncf)
    img = ncd['Sigma0_VV_db']  # (Sentinel-1 band name after db conversion)
    h, w = img.shape

    # # debugging blocks
    # fig, ax = plt.subplots()
    # ax.imshow(img)
    # for c in CATEGORIES:
    #     if c['has_coords']:
    #         for yc, xc in zip(c['coords'][:, 0], c['coords'][:, 1]):
    #             y0, x0 = (i-HWS for i in (yc, xc))
    #             print(y0, x0)
    #             rect = patches.Rectangle((x0, y0), WS, WS,
    #                                      linewidth=1, edgecolor=c['color'],
    #                                      facecolor='none')
    #             ax.add_patch(rect)
    #             ax.scatter(xc, yc, color=c['color'])
    # plt.show()

    # getting total number of coordinates
    ncoords = np.sum([c['coords'].shape[0]
                      for c in CATEGORIES
                      if c['has_coords']])

    # saving block labels
    labels_filename = join(CLASSIFICATION_FEATURES_OUTPUT,
                           f'{base}_labels.bin')
    categories = discarray(labels_filename, 'w+', np.int32, ncoords)
    cur = 0
    for C, c in enumerate(CATEGORIES):
        if c['has_coords']:
            n = c['coords'].shape[0]
            categories[cur:cur+n] = C
            cur += n

    # saving block features
    for feature in FEATURES:
        feature_filename = join(CLASSIFICATION_FEATURES_OUTPUT,
                                f"{base}_{feature['name']}.bin")
        feature['values'] = discarray(
            feature_filename, 'w+', np.float64, ncoords)

    cur = 0
    for c in CATEGORIES:
        if c['has_coords']:
            # defining block
            for yc, xc in zip(c['coords'][:, 0],
                              c['coords'][:, 1]):
                yi, xi = yc-HWS, xc-HWS
                yf, xf = yi+WS, xi+WS
                # checking if block coordinates are inside array boundaries
                inbounds = (0 <= yi <= yf <= h and 0 <= xi <= xf <= h)

                for feature in FEATURES:
                    if inbounds:
                        block = img[yi:yf, xi:xf]
                        feature_val = feature['function'](block)
                    else:
                        feature_val = np.nan
                feature["values"][cur] = feature_val
                cur += 1

    # saving blocks themselves
    cur = 0
    for c in CATEGORIES:
        if c['has_coords']:
            # defining block
            for yc, xc in zip(c['coords'][:, 0],
                              c['coords'][:, 1]):
                yi, xi = yc-HWS, xc-HWS
                yf, xf = yi+WS, xi+WS
                # checking if block coordinates are inside array boundaries
                inbounds = (0 <= yi <= yf <= h and 0 <= xi <= xf <= h)
                if inbounds:
                    block_filename = join(CLASSIFICATION_BLOCKS_OUTPUT,
                                          f"{base}_block_{cur}.bin")
                    block_file = discarray(block_filename, 'w+', np.float64,
                                           block.shape)
                    block = img[yi:yf, xi:xf]
                    block_file[...] = block
                cur += 1
