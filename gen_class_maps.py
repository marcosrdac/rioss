from datetime import datetime
from os import listdir, mkdir
from os.path import expanduser, basename,  exists, isfile, isdir, relpath, dirname, join, splitext
from random import shuffle
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import seaborn as sns
from routines.functions import discarray, get_mwa, get_mrwa, mwsd
from routines.overpool import overlapping_pool
# from classification import classify
# from adth_classification import classify
# from new_adth_classification import classify
from classification_rf_2 import classify


bands_used = {
    'Sigma0_IW1_VV_db',
    'Sigma0_IW2_VV_db',
    'Intensity_IW2_VV_db',
    'Sigma0_VV_db',
    'Sigma0_db',
}


IN = '/mnt/hdd/home/tmp/los/data/original'
TODAY = datetime.today().strftime('%Y%m%d')
OUT = f'/mnt/hdd/home/tmp/los/data/maps/{TODAY}_cls_maps'
BIN = join(OUT, 'bin')
IMG = join(OUT, 'img')
for p in [OUT, BIN, IMG]:
    if not exists(p):
        mkdir(p)


files = [join(IN, f) for f in listdir(IN)]
shuffle(files)


# DISCARRAYS
# f = join(INPUT,
# 'subset_1_of_S1A_IW_SLC__1SDV_20170709T000134_20170709T000201_017387_01D0AA_205D.nc')
# # print(ncd.variables.keys())

# img = discarray(f'/tmp/{basename(f)}', mode="w+", shape=_img.shape)
# img[...] = _img[...]


for f in files:
    name = splitext(basename(f))[0]

    ncd = nc.Dataset(f)
    for band in bands_used:
        try:
            img = ncd[band]
            break
        except:
            continue

    fig, axes = plt.subplots(1,2)
    # fig.suptitle(name)
    ax = axes.flat[0]
    im = ax.imshow(img, aspect="equal")
    plt.colorbar(im, ax=ax)
    # pool = overlapping_pool(img, 512//2, lambda x: 1)
    pool = overlapping_pool(img, 512//2, classify)
    _pool = discarray(join(BIN, name + '.bin'), mode="w+",
                      dtype=int, shape=pool.shape)
    _pool[...] = pool[...]
    # np.save(join(BIN, name + '.bin'), pool)
    ax = axes.flat[1]
    # sns.heatmap(pool, annot=True, square=True)
    sns.heatmap(pool, annot=False, square=True, ax=ax, vmin=0, vmax=1)
    plt.savefig(join(IMG, name + '.png'))
    # plt.show()
