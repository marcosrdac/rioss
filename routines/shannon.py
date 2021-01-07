from os import listdir
from os.path import basename, splitext, join, realpath, expanduser

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.stats import entropy as sequence_entropy
from numba import jit, prange
from tqdm import tqdm
from PIL import Image

from .functions import timer, normalize01


def validate_index(values, x=None):
    if x is None:
        x = np.arange(1, values.size+1)
    else:
        x = np.array(x)
    values = np.array(values)
    index = np.logical_and(x > 0, values > 0)
    return(index)


def entropy(img, base=None):
    if isinstance(img, np.ma.MaskedArray):
        img = img.compressed()
    entropy = sequence_entropy(img.flat, base=base)
    return entropy

def norm_entropy(img):
    return entropy(img.ravel())/np.log(img.size)

def files_entropy(filepaths, base=None):
    entropies = []
    for i, filepath in enumerate(filepaths):
        filepath = expanduser(filepath)
        img = np.mean(np.array(Image.open(filepath)), axis=-1)
        #img = ndimage.gaussian_filter(img, sigma=1)
        #img = normalize01(img)
        entropies.append(entropy(img, base=base))
    return(entropies)


def files_box_counting_plot(filepaths, ax=None, fig=None):  # not using args
    fig, ax = plt.subplots()

    for i, filepath in enumerate(filepaths):
        filepath = expanduser(filepath)
        name = basename(splitext(filepath)[0])
        img = np.mean(np.array(Image.open(filepath)), axis=-1)
        # plt.imshow(img)
        # plt.show()
        x = np.linspace(0.1, .9, 10)
        y = np.empty_like(x)
        for i, p in enumerate(x):
            y[i] = fractal_dimension(img, p)
        plt.plot(x, y)
        #ax.hist(img.flat, )
    fig.legend()
    plt.show()


if __name__ == '__main__':
    folder = 'pics'
    filepaths = [join(folder, i) for i in listdir(folder)]
    filepaths.sort()

    entropies = []
    for filepath in filepaths:
        e = files_entropy([filepath])[0]
        entropies.append(e)
        print(f"{e:3f} - {basename(filepath)}")
    print()

    import seaborn as sns
    sns.heatmap(np.array(entropies).reshape(len(entropies), 1)[:-3]-15.2,
                annot=True)
    plt.show()
