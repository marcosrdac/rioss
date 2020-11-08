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
from routines.overpool.overpool import overlapping_pool as overpool
# from classification import classify


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
    ncd = nc.Dataset(f)
    for b in bands_used:
        if b in ncd.variables:
            band = b
            img = ncd[b]
            minimum, maximum = np.min(img), np.max(img)
            print(basename(f))
            print(np.shape(img))
            # print(f"{minimum}\t{maximum}\t{band}")
            # print(img)
            break

    #plt.subplot(121)
    ## plt.suptitle(basename(f))
    #plt.imshow(img, aspect="equal")
    #plt.subplot(122)
    ## plt.suptitle(basename(f))
    #pool = overpool(img, 512, classify)
    #_pool = discarray(f'{basename(f)}.bin', mode="w+",
    #                  dtype=int, shape=pool.shape)
    #_pool[...] = pool[...]
    #sns.heatmap(pool, annot=True, square=True)
    #plt.savefig(f'{basename(f)}.png')
    #plt.show()
