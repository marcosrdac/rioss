from os import listdir
from os.path import basename, splitext, join, realpath, expanduser

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import semivariogram
import psd
import boxcounting
from utils import normalize01


if __name__ == '__main__':
    folder = 'pics'
    filepaths = [join(folder, i) for i in listdir(folder)]
    filepaths.sort()

    boxcounting.files_box_counting_plot(filepaths)

    print('comecando')
    boxcounting_Ds = boxcounting.files_fractal_dimension(filepaths, 0.5)
    print('box')
    psd_Ds = psd.files_fractal_dimension(filepaths)
    print('psd')
    semivariogram_Ds = semivariogram.files_fractal_dimension(filepaths, .01/100)
    print('semi')
    im = np.array([boxcounting_Ds, psd_Ds, semivariogram_Ds])
    #for i in range(3):
    #    im[i, :] = normalize01(im[i, :])
    # plt.imshow(im)
    fig, ax = plt.subplots()
    sns.heatmap(im, annot=True, ax=ax)
    ax.set_xticklabels(filepaths)
    ax.set_yticklabels(['box counting', 'power spectrum', 'semivariogram'])
    #sns.heatmap(np.corrcoef(im), annot=True)
    plt.show()
    # files_psd_plot(filepaths)
