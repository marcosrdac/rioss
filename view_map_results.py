from os import environ, listdir, mkdir
from os.path import expanduser, basename,  exists, isfile, isdir, relpath, dirname, join, splitext
from scipy.ndimage import gaussian_filter, laplace
from parameters import CLASSIFICATION_CATEGORIES
from routines.functions import discarray, unsigned_span, grad, grad_max, grad_mean, grad_median, lap, lap_max
import cv2 as cv
import numpy as np
import netCDF4 as nc
import matplotlib as mpl
import matplotlib.pyplot as plt
from os.path import join
from os import listdir, environ
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

MAPS = "/mnt/hdd/home/tmp/los/data/maps/2021020318_7_class_maps/bin"
ORIGINAL = "/home/marcosrdac/tmp/los/data/original"
ls = [join(MAPS, f) for f in listdir(MAPS)]


colors = [c['color'] for c in CLASSIFICATION_CATEGORIES]
labels = [c['name'] for c in CLASSIFICATION_CATEGORIES]
CMAP1 = mpl.colors.LinearSegmentedColormap.from_list('feat_colors', colors)
CMAP1 = mpl.cm.get_cmap(CMAP1, 7)

band_choices = [
    'Sigma0_IW1_VV_db',
    'Sigma0_IW2_VV_db',
    'Intensity_IW2_VV_db',
    'Sigma0_VV_db',
    'Sigma0_db',
]

im = plt.imshow([np.arange(7)], cmap=CMAP1)
plt.close()

i=0
for f in ls:
    i+=1
    if i==1: continue

    name = splitext(basename(f))[0]
    ncf = join(ORIGINAL, name + '.nc')

    ncd = nc.Dataset(ncf)
    for band in band_choices:
        if band in ncd.variables:
            img = np.asarray(ncd.variables[band])
            break

    class_map = discarray(f).astype(int)


    fig, axes = plt.subplots(1, 2)
    ax0, ax1 = axes

    # im0 = ax0.pcolormesh(img[::8, ::8])
    # ax0.set_title()
    # plt.colorbar(im0, ax=ax0)

    ax1.set_title('Classification map')

    boundaries = np.arange(0, len(labels)+1)
    values = boundaries[:-1]
    ticks = values + 0.5

    levels = np.arange(7)
    # norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    pc1 = ax1.imshow(class_map, cmap=CMAP1, vmin=0, vmax=6)

    cbar1 = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, orientation='horizontal',
                         boundaries=boundaries, values=values, ticks=boundaries)

    cbar1.set_ticks(ticks)
    cbar1.set_ticklabels(np.arange(len(labels)))
    cbar1.set_ticklabels(labels)

    plt.show()

    i += 1
