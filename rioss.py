from datetime import datetime
from os import listdir, mkdir
from os.path import join, splitext, basename, exists
from routines.functions import discarray
from routines.overpool import overlapping_pool
from classification import apply_detector, apply_classifier
from parameters import CLASSIFICATION_CATEGORIES, CLASSIFICATION_CATEGORIES_2
from parameters import DATA, CLASSIFICATION_INPUT_DATA
import matplotlib as mpl
import numpy as np
import scipy.ndimage as sciimg
import scipy.interpolate as sciint
import netCDF4 as nc
import pandas as pd
import seaborn as sns
import matplotlib as mpl
mpl.use('Agg')
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

def send_warning_email():
    pass


PROBA_THRESHOLD = .1


PROBA = True  # oil probability?
N_CLASSES = 2
# PROBA = False  # oil probability?
# N_CLASSES = 7

VERBOSE = True

CMAP0 = 'gray'
FACTOR = 8
SHOW = False
OUT_EXTS = ['png']  #  ['svg', 'png']
WS = 512
WHS = WS//2
PROJECTION = crs.PlateCarree()
TODAY = datetime.now().strftime('%Y%m%d%H')


# Possible NetCDF4 variables
band_choices = ['Sigma0_IW1_VV_db', 'Sigma0_IW2_VV_db', 'Intensity_IW2_VV_db',
                'Sigma0_VV_db', 'Sigma0_db',]
lat_choices = ['latitude', 'lat', 'LATITUDE', 'LAT',]
lon_choices = {'longitude', 'lon', 'LONGITUDE', 'LON',}


if PROBA:
    MAP_KIND = 'proba'
    MAP_DTYPE = np.float64
    INTERP_METHOD = 'cubic'
    # CMAP1 = 'Reds'
    CMAP1 = 'jet'
    def apply_model(img, proba=PROBA):
        return apply_detector(img, proba, verbose=VERBOSE)[0]
else:
    MAP_KIND = 'class'
    MAP_DTYPE = np.int32
    INTERP_METHOD = 'nearest'
    if N_CLASSES == 2:
        colors = [c['color'] for c in CLASSIFICATION_CATEGORIES if c in CLASSIFICATION_CATEGORIES_2]
        labels = [c['name'].capitalize() for c in CLASSIFICATION_CATEGORIES if c in CLASSIFICATION_CATEGORIES_2]
        def apply_model(img, proba=PROBA, verbose=VERBOSE):
            return apply_detector(img, proba)[0]
    else:
        colors = [c['color'] for c in CLASSIFICATION_CATEGORIES]
        labels = [c['name'].capitalize() for c in CLASSIFICATION_CATEGORIES]
        def apply_model(img, proba=PROBA, verbose=VERBOSE):
            return apply_classifier(img, proba)[0]
    CMAP1 = mpl.colors.LinearSegmentedColormap.from_list('feat_colors', colors)
    IM = plt.imshow([np.arange(7)], cmap=CMAP1)
    plt.close()

# debugging
# def apply_model(img, proba=True):
    # return 1


# Input folder settings
IN = CLASSIFICATION_INPUT_DATA
# Output folder settings
BASE = join(DATA, 'maps')
OUT = join(BASE, f'{TODAY}_rioss')
BIN = join(OUT, 'bin')
IMG = join(OUT, 'img')
RESULTS = join(OUT, 'results')

for p in [OUT, BIN, IMG, RESULTS]:
    if not exists(p): mkdir(p)


ls = [join(IN, f) for f in listdir(IN) if f.endswith('.nc')]

for f in ls:
    name = splitext(basename(f))[0]
    
    of = join(IN, name + ".nc")
    ncd = nc.Dataset(of)

    for lat in lat_choices:
        if lat in ncd.variables:
            lat = ncd.variables[lat]
            break
        else:
            lat = None

    for lon in lon_choices:
        if lon in ncd.variables:
            lon = ncd.variables[lon]
            break
        else:
            lon = None

    for band in band_choices:
        if band in ncd.variables:
            img = ncd.variables[band]
    # img = discarray('/home/marcosrdac/article_los/campos/interpolated.bin', dtype=np.float32)

    # print(np.asarray(img).size)
    if img.size < 1024**2:
        print(f'*** Skipping {name} for it is too small.')
        continue

    try:
        raise Exception
        mask = img[:].mask
        lon_mask = mask[mask.shape[0]//2, :]
        lat_mask = mask[:, mask.shape[1]//2]
        min_lon, max_lon = (np.where(~lon_mask)[0][[0,-1]])
        min_lat, max_lat = (np.where(~lat_mask)[0][[0,-1]])

        min_lon_lat_mask = mask[:, min_lon]
        max_lon_lat_mask = mask[:, max_lon]
        min_lat_lon_mask = mask[ min_lat, :]
        max_lat_lon_mask = mask[ max_lat, :]
        min_lon_min_lat, min_lon_max_lat = (np.where(~min_lon_lat_mask)[0][[0,-1]])
        max_lon_min_lat, max_lon_max_lat = (np.where(~max_lon_lat_mask)[0][[0,-1]])
        min_lat_min_lon, min_lat_max_lon = (np.where(~min_lat_lon_mask)[0][[0,-1]])
        max_lat_min_lon, max_lat_max_lon = (np.where(~max_lat_lon_mask)[0][[0,-1]])

        max_min_lon = np.max([min_lat_min_lon, max_lat_min_lon])
        min_max_lon = np.min([min_lat_max_lon, max_lat_max_lon])
        max_min_lat = np.max([min_lon_min_lat, max_lon_min_lat])
        min_max_lat = np.min([min_lon_max_lat, max_lon_max_lat])

        min_lat = max_min_lat
        max_lat = min_max_lat
        min_lon = max_min_lon
        max_lon = min_max_lon

        del min_lon_lat_mask, max_lon_lat_mask, min_lat_lon_mask, max_lat_lon_mask

    except:
        min_lon, max_lon = 0, img.shape[0]
        min_lat, max_lat = 0, img.shape[1]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10,7), dpi=300, sharex=True,
                                   subplot_kw={'projection': PROJECTION})

    slon = lon[min_lat:max_lat:FACTOR, min_lon:max_lon:FACTOR]
    slat = lat[min_lat:max_lat:FACTOR, min_lon:max_lon:FACTOR]
    simg = img[min_lat:max_lat:FACTOR, min_lon:max_lon:FACTOR]
    # simg[simg.mask] = np.nan
    # simg = sciimg.gaussian_filter(simg, .5)

    pc0 = ax0.pcolormesh(slon, slat, simg, shading='nearest', cmap=CMAP0)
    gl0 = ax0.gridlines(crs=PROJECTION, draw_labels=True,
                  linewidth=1, color='white', alpha=.2, linestyle='--')
    gl0.top_labels=False
    gl0.right_labels=False
    cbar0 = plt.colorbar(pc0, ax=ax0, fraction=0.046, pad=0.04, orientation='horizontal')
    # cbar0.ax.set_xlabel('$\sigma_0$', rotation=0)
    ax0.set_title('$\sigma_0\ (dB)$')

    def mid(subimg, window=None):
        return np.mean(window, axis=1)

    wimg = img[min_lat:max_lat, min_lon:max_lon]

    out = overlapping_pool(wimg, whs=WHS, pool_func=apply_model, extra=False, give_window=False, dtype=MAP_DTYPE)
    # out = discarray('/mnt/hdd/home/tmp/los/data/maps/2021030510_2_proba_maps/bin/1_S1A_IW_SL1C__1SDV_20210125T095454_20210125T095521_036293_044201_379E_split_Orb_Cal_deb_ML_dB.bin', dtype=MAP_DTYPE)

    _out = discarray(join(BIN, name + '.bin'), mode='w+', shape=out.shape, dtype=out.dtype)
    _out[...] = out

    idx = overlapping_pool(wimg, whs=WHS, pool_func=mid, extra=False, give_window=True, last_dim=2, dtype=int)
    y = idx[...,0].ravel() + min_lat
    x = idx[...,1].ravel() + min_lon
    out_lat = lat[:][y, x].reshape(out.shape)
    out_lon = lon[:][y, x].reshape(out.shape)

    out_coords = np.stack([out_lat.ravel(), out_lon.ravel()], axis=1)
    out_values = out.ravel()
    # print(slon.shape)
    # print(slat.shape)
    sout = sciint.griddata(out_coords, out_values, (slat, slon), method=INTERP_METHOD)
    # print(sout.shape)

    at_risk = out_values >= PROBA_THRESHOLD
    x_risk, y_risk = x[at_risk], x[at_risk]
    lat_risk, lon_risk = out_lat.ravel()[at_risk], out_lon.ravel()[at_risk]
    out_risk = out.ravel()[at_risk]

    if np.any(at_risk):
        sheet = np.stack([x_risk, y_risk, lat_risk, lon_risk, out_risk], axis=1)
        np.savetxt(join(RESULTS, name + '.csv'), sheet, header='x,y,lat,lon,proba')
        # send_warning_email()

    # cmap configuration
    if PROBA:
        pc1 = ax1.pcolormesh(slon, slat, sout, shading='nearest', cmap=CMAP1, vmin=0, vmax=1)
        ax1.set_title('Oil probability')
        cbar1 = plt.colorbar(pc1, ax=ax1, fraction=0.046, pad=0.04, orientation='horizontal')
    else:
        pc1 = ax1.pcolormesh(slon, slat, sout, shading='nearest', cmap=CMAP1, vmin=0, vmax=N_CLASSES-1)
        ax1.set_title('Classification map')
        boundaries = np.arange(0, len(labels)+1)
        values = boundaries[:-1]
        ticks = values + 0.5
        cbar1 = plt.colorbar(IM, ax=ax1, fraction=0.046, pad=0.04, orientation='horizontal',
                            boundaries=boundaries, values=values, ticks=boundaries)
        cbar1.set_ticks(ticks)
        cbar1.set_ticklabels(np.arange(len(labels)))
        cbar1.set_ticklabels(labels)


    gl1 = ax1.gridlines(crs=PROJECTION, draw_labels=True,
                  linewidth=1, color='white', alpha=.2, linestyle='--')
    gl1.top_labels=False
    gl1.right_labels=False
    # ax1.set_xlim(ax0.get_xlim())
    # ax1.set_ylim(ax0.get_ylim())

    # break

    # plt.tight_layout()
    for ext in OUT_EXTS:
        plt.savefig(join(IMG, name + '.' + ext))

    if SHOW:
        plt.show()
    plt.close(fig)
    # ncd.close()
