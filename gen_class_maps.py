from os import listdir, makedirs
from os.path import expanduser, isdir, splitext, join, basename
from random import shuffle
import numpy as np
import netCDF4 as nc
from keras.models import load_model  # needs to be imported here
from pickle import loads
import matplotlib.pyplot as plt
import seaborn as sns
from routines.functions import discarray, get_mwa, get_mrwa, mwsd
from routines.overpool import overlapping_pool
# from classification import classify
# from adth_classification import classify
# from new_adth_classification import classify
from rf_2_classification import classify


bands_used = {
    'Sigma0_IW1_VV_db',
    'Sigma0_IW2_VV_db',
    'Intensity_IW2_VV_db',
    'Sigma0_VV_db',
    'Sigma0_db',
}


folder = '/mnt/hdd/home/tmp/los/data/original/'

files = [join(folder, f) for f in listdir(folder)]
shuffle(files)
for f in files:

    # f = join(folder,
    # 'subset_1_of_S1A_IW_SLC__1SDV_20170709T000134_20170709T000201_017387_01D0AA_205D.nc')
    # # print(ncd.variables.keys())

    # img = discarray(f'/tmp/{basename(f)}', mode="w+", shape=_img.shape)
    # img[...] = _img[...]

    ncd = nc.Dataset(f)
    for band in bands_used:
        try:
            img = ncd[band]
            break
        except:
            continue
    fig, axes = plt.subplots(1,2)
    # fig.suptitle(basename(f))
    ax = axes.flat[0]
    im = ax.imshow(img, aspect="equal")
    plt.colorbar(im, ax=ax)
    # pool = overlapping_pool(img, 512//2, lambda x: 1)
    pool = overlapping_pool(img, 512//2, classify)
    _pool = discarray(f'out/{basename(f)}.bin', mode="w+",
                      dtype=int, shape=pool.shape)
    _pool[...] = pool[...]
    ax = axes.flat[1]
    # sns.heatmap(pool, annot=True, square=True)
    sns.heatmap(pool, annot=False, square=True, ax=ax, vmin=0, vmax=1)
    plt.savefig(f'out/{basename(f)}.png')
    # plt.show()
