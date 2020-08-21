from os import listdir
from os.path import basename, splitext, join, realpath, expanduser
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
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
    index = np.logical_and(x > 0, values > 0).astype(int)
    return(index)


# Modified from:
#   https://gist.github.com/viveksck/1110dfca01e4ec2c608515f0d5a5b1d1
def fractal_dimension(img, threshold=0.5):
    '''
    Calculates Minkowski–Bouligand fractal dimension of a 2D array.
    '''

    def boxcount(img, k):
        S = np.add.reduceat(
                np.add.reduceat(img, np.arange(0, img.shape[0], k), axis=0),
                            np.arange(0, img.shape[1], k), axis=1)
        # We count non-empty (0) and non-full boxes (k*k)
        return len(np.where((S > 0) & (S < k*k))[0])

    img = normalize01(img)

    # Only for 2d image
    assert(len(img.shape) == 2)

    # Transform img into a binary array
    img = (img < threshold)

    # Minimal dimension of image
    p = min(img.shape)

    # Greatest power of 2 less than or equal to p
    n = 2**np.floor(np.log(p)/np.log(2))

    # Extract the exponent
    n = int(np.log(n)/np.log(2))

    # Build successive box sizes (from 2**n down to 2**1)
    sizes = 2**np.arange(n, 1, -1)

    # Actual box counting with decreasing size
    counts = []
    for size in sizes:
        counts.append(boxcount(img, size))

    counts = np.array(counts)
    sizes = np.array(sizes)
    # plt.plot(sizes, counts)
    # plt.show()

    # Fit the successive log(sizes) with log (counts)
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return(-coeffs[0])


def files_fractal_dimension(filepaths, threshold):
    fractal_dimensions = []
    for i, filepath in enumerate(filepaths):
        filepath = expanduser(filepath)
        name = basename(splitext(filepath)[0])
        img = np.mean(np.array(Image.open(filepath)), axis=-1)
        #img = ndimage.gaussian_filter(img, sigma=1)
        img = normalize01(img)
        fractal_dimensions.append(fractal_dimension(img, threshold))
    return(fractal_dimensions)


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

    files_box_counting_plot(filepaths)
