from datetime import datetime
from os import listdir, mkdir
from os.path import join, splitext, basename, exists
from routines.functions import discarray, get_mwa, get_mrwa, mwsd
from routines.overpool import overlapping_pool
# from classification_rf_2 import classify
from parameters import CLASSIFICATION_CATEGORIES, CLASSIFICATION_CATEGORIES_2
import matplotlib as mpl
import numpy as np
import scipy.ndimage as sciimg
import scipy.interpolate as sciint
import netCDF4 as nc
import pandas as pd
import seaborn as sns
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as mplticker
import cartopy.crs as crs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from shapely.vectorized import contains
plt.style.use('seaborn')
sns.set_context('paper')


N_CLASSES = 7
CMAP0 = 'gray'
CMAP1 = 'jet'
OUT_EXT = 'png'
WS = 512
WHS = WS//2
TODAY = datetime.now().strftime('%Y%m%d%H')
DTYPE = np.float64


if N_CLASSES == 2:
    from classification_rf_2 import FEATURES, calculate_features
else:                                         
    from classification_rf_7 import FEATURES, calculate_features

print([f['name'] for f in FEATURES])
exit()

# Possible NetCDF4 variables
band_choices = ['Sigma0_IW1_VV_db', 'Sigma0_IW2_VV_db', 'Intensity_IW2_VV_db',
                'Sigma0_VV_db', 'Sigma0_db',]

# Input folder settings
IN = '/home/marcosrdac/tmp/los/data/local'
# Output folder settings
BASE = '/mnt/hdd/home/tmp/los/data/maps'
OUT = join(BASE, f'{TODAY}_{N_CLASSES}_feats')
BIN = join(OUT, 'bin')
IMG = join(OUT, 'img')


for p in [OUT, BIN, IMG]:
    if not exists(p): mkdir(p)

ls = [join(IN, f) for f in listdir(IN) if f.endswith('.nc')]

for f in ls:
    name = splitext(basename(f))[0]
    
    of = join(IN, name + ".nc")
    ncd = nc.Dataset(of)

    for band in band_choices:
        if band in ncd.variables:
            img = ncd.variables[band]

    out = overlapping_pool(img, whs=WHS, pool_func=calculate_features, extra=False, give_window=False, dtype=DTYPE, last_dim=len(FEATURES))
    # out = discarray(join(BIN, name + '.bin'), mode='r', dtype=DTYPE)
    mean = np.mean(out, axis=(0, 1))

    _out = discarray(join(BIN, name + '.bin'), mode='w+', shape=out.shape, dtype=out.dtype)
    _out[...] = out
    _mean = discarray(join(BIN, name + '_mean' + '.bin'), mode='w+', shape=mean.shape, dtype=mean.dtype)
    _mean[...] = mean

    print(mean)
    print(mean.shape)

    fig, axes = plt.subplots(1+len(FEATURES), 1, figsize=(4,4+4*len(FEATURES)))

    axes.flat[0].imshow(img)
    for i in range(len(FEATURES)):
        axes.flat[1+i].set_title([feat['name'] for feat in FEATURES][i])
        axes.flat[1+i].imshow(_out[:,:,i])
    for ax in axes.flat:
        ax.grid(False)

    fig.savefig(join(IMG, name + '.png'))
    # plt.show()
