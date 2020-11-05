#!/usr/bin/env python


# INTERNAL DEPENDENCIES
import re
from os.path import splitext, join, basename
# EXTERNAL DEPENDENCIES
import numpy as np
import netCDF4 as nc
from PIL import Image
# DEFINED
from routines.functions import discarray, listifext, first_channel, \
    sample_min_dist, get_block_corners, adjust_block_center_get_corners
from segmentation import segmentate
from parameters import CLASSIFICATION_INPUT_DATA, \
    CLASSIFICATION_INPUT_MASKS, \
    CLASSIFICATION_BLOCKS_OUTPUT, \
    CLASSIFICATION_FEATURES_OUTPUT, \
    CLASSIFICATION_CATEGORIES, \
    CLASSIFICATION_FEATURES
# DEBUGGING
#import matplotlib.pyplot as plt
#import matplotlib.patches as patches


# PRECONFIGURATION

# block side, half-side
WS = 512
HWS = WS//2


input_data_fps = listifext(CLASSIFICATION_INPUT_DATA, 'nc', fullpath=True)
input_masks_fps = listifext(CLASSIFICATION_INPUT_MASKS, 'jpg', fullpath=True)


bands_used = {
    'Sigma0_IW1_VV_db',
    'Sigma0_IW2_VV_db',
    'Intensity_IW2_VV_db',
    'Sigma0_VV_db',
    'Sigma0_db',
}


for ncf in input_data_fps:
    print(f'Using file: "{ncf}".')
    print('  Mask files found:')
    # getting mask data
    base = splitext(basename(ncf))[0]
    mask_files = [f for f in input_masks_fps if base in f]

    for c in CLASSIFICATION_CATEGORIES:
        c['coords'] = []
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
            for c in CLASSIFICATION_CATEGORIES:
                if mask_name == c['name']:
                    c['has_coords'] = True
                    c['coords'] = sampled_mask_coords
                    break
        else:
            print(f'    [ WARNING ] No points in "{mask_name}".')

    # if no coordinates for masks are achieved
    if not any([c['has_coords'] for c in CLASSIFICATION_CATEGORIES]):
        print(
            f'    None\n'
            f'Please add its masks to "{CLASSIFICATION_INPUT_MASKS}".\n'
            f'Skipping image.')
        continue
 
    # getting total number of coordinates
    ncoords = np.sum([c['coords'].shape[0]
                      for c in CLASSIFICATION_CATEGORIES
                      if c['has_coords']])
 
    # opening original data
    ncd = nc.Dataset(ncf)
    for band in bands_used:
        try:
            img = ncd[band]
            break
        except:
            continue
 
    h, w = img.shape
 
 ##    # --- debugging blocks (looking at their positions) --- #
 ##    fig, ax = plt.subplots()
 ##    ax.imshow(img)
 ##    for c in CLASSIFICATION_CATEGORIES:
 ##        if c['has_coords']:
 ##            for p, (yc, xc) in enumerate(zip(c['coords'][:, 0],
 ##                                             c['coords'][:, 1])):
 ##                modified, (yc, xc), (yi, xi), (yf, xf) = \
 ##                    adjust_block_center_get_corners(yc, xc, WS, h, w)
 ##                if modified:
 ##                    c['coords'][p, :] = (yc, xc)
 ##                rect = patches.Rectangle((xi, yi), WS, WS,
 ##                                         linewidth=1, edgecolor=c['color'],
 ##                                         facecolor='none')
 ##                ax.add_patch(rect)
 ##                ax.scatter(xc, yc, color=c['color'])
 ##    plt.show()
  
    # --- saving block labels (comment if not needed) --- #
    labels_filename = join(CLASSIFICATION_FEATURES_OUTPUT,
                           f'{base}_label.bin')

    categories = discarray(labels_filename, 'w+', np.int64, ncoords)
    cur = 0
    for C, c in enumerate(CLASSIFICATION_CATEGORIES):
        if c['has_coords']:
            n = c['coords'].shape[0]
            categories[cur:cur+n] = C
            cur += n

    # --- saving block features (comment if not needed) --- #
    print(f'  Saving features for all blocks in image "{base}"')
    for feature in CLASSIFICATION_FEATURES:
        feature_filename = join(CLASSIFICATION_FEATURES_OUTPUT,
                                f"{base}_{feature['abbrv']}.bin")
        feature['values'] = discarray(
            feature_filename, 'w+', np.float64, ncoords)

    cur = 0
    for c in CLASSIFICATION_CATEGORIES:
        if c['has_coords']:
            # defining block
            for p, (yc, xc) in enumerate(zip(c['coords'][:, 0],
                                             c['coords'][:, 1])):
                print(f'  Block centered at y={yc}, x={xc}')

                modified, (yc, xc), (yi, xi), (yf, xf) = \
                    adjust_block_center_get_corners(yc, xc, WS, h, w)
                if modified:
                    c['coords'][p, :] = (yc, xc)
                    print(f'  Adjusting block center to y={yc}, x={xc}')

                block = img[yi:yf, xi:xf]
                segmented = segmentate(block)

                for feature in CLASSIFICATION_FEATURES:
                    print(f'    Calculating feature: {feature["name"]}')
                    feature_val = feature['function'](block, segmented)
                    feature["values"][cur] = feature_val
                print()
                cur += 1

#    # --- saving blocks themselves (comment if not needed) --- #
#    cur = 0
#    for c in CLASSIFICATION_CATEGORIES:
#        if c['has_coords']:
#            # defining block
#            for yc, xc in zip(c['coords'][:, 0],
#                              c['coords'][:, 1]):
#                (yi, xi), (yf, xf) = get_block_corners(yc, xc, WS)
#                block_filename = join(CLASSIFICATION_BLOCKS_OUTPUT,
#                                      f"{base}_block_{cur}.bin")
#                block_file = discarray(block_filename, 'w+', np.float64,
#                                       block.shape)
#                block = img[yi:yf, xi:xf]
#                block_file[...] = block
#                cur += 1
