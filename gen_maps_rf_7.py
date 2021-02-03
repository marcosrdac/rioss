from datetime import datetime
from os import listdir, mkdir
from os.path import join, splitext, basename, exists
from routines.functions import discarray, get_mwa, get_mrwa, mwsd
from routines.overpool import overlapping_pool
from classification_rf_2 import classify
import matplotlib as mpl
import numpy as np
import scipy.ndimage as sciimg
import scipy.interpolate as sciint
import netCDF4 as nc
import pandas as pd
import seaborn as sns
import matplotlib as mpl
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

proba = True  # oil probability?
cmap0 = 'gray'
factor = 8
show = False
out_ext = 'png'
# factor = 100

IN = '/mnt/hdd/home/tmp/los/data/original'
# IN = '/home/marcosrdac/tmp/los/data/test_cases'

TODAY = datetime.today().strftime('%Y%m%d')
if proba:
    OUT = f'/mnt/hdd/home/tmp/los/data/maps/{TODAY}_prob_maps'
    cmap1 = 'Reds'
    def _classify(img, proba=True):
        return classify(img, proba)[0]
else:
    OUT = f'/mnt/hdd/home/tmp/los/data/maps/{TODAY}_cls_maps'
    cmap1 = 'Greys_r'
    def _classify(img, proba=False):
        return classify(img, proba)[0]
BIN = join(OUT, 'bin')
IMG = join(OUT, 'img')
for p in [OUT, BIN, IMG]:
    if not exists(p):
        mkdir(p)

# def _classify(img, proba=True):
    # return 1

ls = [join(IN, f) for f in listdir(IN) if f.endswith('.nc')]

band_choices = {
    'Sigma0_IW1_VV_db',
    'Sigma0_IW2_VV_db',
    'Intensity_IW2_VV_db',
    'Sigma0_VV_db',
    'Sigma0_db',
}

lat_choices = ['latitude', 'lat', 'LATITUDE', 'LAT',]
lon_choices = {'longitude', 'lon', 'LONGITUDE', 'LON',}

ws = 512
whs = ws//2

# i=0
for f in ls:
    # i += 1
    # if i ==1: continue
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
    
    if img.size < 1024**2:
        print(f'*** Skipping {name} for it is too small.')
        continue

    # img = discarray('/home/marcosrdac/article_los/campos/interpolated.bin', dtype=np.float32)

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

    projection = crs.PlateCarree()

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10,7), dpi=300, sharex=True,
                                   subplot_kw={'projection': projection})

    slon = lon[min_lat:max_lat:factor, min_lon:max_lon:factor]
    slat = lat[min_lat:max_lat:factor, min_lon:max_lon:factor]
    simg = img[min_lat:max_lat:factor, min_lon:max_lon:factor]
    # simg[simg.mask] = np.nan
    # simg = sciimg.gaussian_filter(simg, .5)

    # ax0 = fig.add_subplot(211, projection=projection)
    pc0 = ax0.pcolormesh(slon, slat, simg, shading='nearest', cmap=cmap0)
    gl0 = ax0.gridlines(crs=projection, draw_labels=True,
                  linewidth=1, color='white', alpha=.2, linestyle='--')
    gl0.xlabels_top=False
    gl0.ylabels_right=False
    cbar0 = plt.colorbar(pc0, ax=ax0, fraction=0.046, pad=0.04, orientation='horizontal')
    # cbar0.ax.set_xlabel('$\sigma_0$', rotation=0)
    ax0.set_title('$\sigma_0$')



    def mid(subimg, window=None):
        return np.mean(window, axis=1)

    wimg = img[min_lat:max_lat, min_lon:max_lon]

    out = overlapping_pool(wimg, whs=whs, pool_func=_classify, extra=False, give_window=False, dtype=float)
    # out = discarray('/home/marcosrdac/article_los/campos/subset_0_of_S1B_IW_SLC__1SDV_20210120T080527_20210120T080554_025235_030137_BB07_Cal_Orb_deb_ML.bin', dtype=np.float64)
    _out = discarray(join(BIN, name + '.bin'), mode='w+', shape=out.shape, dtype=out.dtype)
    _out[...] = out
    idx = overlapping_pool(wimg, whs=whs, pool_func=mid, extra=False, give_window=True, last_dim=2, dtype=int)
    # del wimg

     
    y = idx[...,0].ravel() + min_lat
    x = idx[...,1].ravel() + min_lon
    out_lat = lat[:][y, x].reshape(out.shape)
    out_lon = lon[:][y, x].reshape(out.shape)

    out_coords = np.stack([out_lat.ravel(), out_lon.ravel()], axis=1)
    out_values = out.ravel()
    print(slon.shape)
    print(slat.shape)
    sout = sciint.griddata(out_coords, out_values, (slat, slon), method='cubic')
    print(sout.shape)


    # ax1 = fig.add_subplot(212, projection=projection)
    # pc1 = ax1.pcolormesh(out_lon, out_lat, out, shading='flat', cmap=cmap1, vmin=0, vmax=1)
    pc1 = ax1.pcolormesh(slon, slat, sout, shading='flat', cmap=cmap1, vmin=0, vmax=1)
    gl1 = ax1.gridlines(crs=projection, draw_labels=True,
                  linewidth=1, color='white', alpha=.2, linestyle='--')
    gl1.xlabels_top=False
    gl1.ylabels_right=False
    cbar1 = plt.colorbar(pc1, ax=ax1, fraction=0.046, pad=0.04, orientation='horizontal')
    # cbar1.ax.set_xlabel('Oil probability', rotation=0)
    ax1.set_title('Oil probability')
    # ax1.set_xlim(ax0.get_xlim())
    # ax1.set_ylim(ax0.get_ylim())

    # break

    # plt.tight_layout()
    plt.savefig(join(IMG, name + '.' + out_ext))
    if show:
        plt.show()
    plt.close(fig)
    ncd.close()
