from os import listdir
from os.path import basename, splitext, join, realpath, expanduser
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from numba import jit, prange
from tqdm import tqdm
from PIL import Image
from utils import timer, normalize01


# From https://github.com/rougier/numpy-100 (#87)
def boxcount(Z, k):
    S = np.add.reduceat(
                np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                np.arange(0, Z.shape[1], k), axis=1)
    # We count non-empty (0) and non-full boxes (k*k)
    return len(np.where((S > 0) & (S < k*k))[0])


# Modified from:
#   https://gist.github.com/viveksck/1110dfca01e4ec2c608515f0d5a5b1d1
def fractal_dimension(Z, threshold=0.9):
    '''
    Calculates Minkowski–Bouligand fractal dimension of a 2D array.
    '''
    Z = normalize01(Z)

    # Only for 2d image
    assert(len(Z.shape) == 2)

    # Transform Z into a binary array
    Z = (Z < threshold)

    # Minimal dimension of image
    p = min(Z.shape)

    # Greatest power of 2 less than or equal to p
    n = 2**np.floor(np.log(p)/np.log(2))

    # Extract the exponent
    n = int(np.log(n)/np.log(2))

    # Build successive box sizes (from 2**n down to 2**1)
    sizes = 2**np.arange(n, 1, -1)

    # Actual box counting with decreasing size
    counts = []
    for size in sizes:
        counts.append(boxcount(Z, size))

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
        fractal_dimensions.append(fractal_dimension(img,.9))
    return(fractal_dimensions)


if __name__ == '__main__':
    #folder='/home/marcosrdac/Dropbox/projects/fractal_dimension/semivariogram_programming/test'
    folder='pics'
    filepaths = [ join(folder, i) for i in listdir(folder) ]
    filepaths.sort()
    #filepaths = ['~/Dropbox/pictures/wallpapers/favorites/boats_happy_sunrise.jpg',]
    box_counting_Ds = files_fractal_dimension(filepaths, 0.5)
    print(box_counting_Ds)
