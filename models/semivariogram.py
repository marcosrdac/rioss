from os import listdir
from os.path import basename, splitext, join, realpath, expanduser
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from numba import jit, prange
from tqdm import tqdm
from PIL import Image
from utils import timer


#@timer
@jit(boundscheck=False, nogil=True, nopython=True, parallel=True)
def semivariogram2d(img, resample=0.):
    '''
    Calculates semivariogram for 2D data.
    :param img: 2D data array.
    :param resample: Samples image pixels before calculating semivariogram.
    :type img: 2D numpy.ndarray object.
    :type resample: integer or percentage float.
    :return: mean semivariance by integer distance (starting
        from 1).
    :rtype: 1D numpy.ndarray.
    '''
    h = img.shape[0]
    w = img.shape[1]
    s = img.size
    z = img.flat
    maxdist = int(round(np.sqrt(h**2+w**2)))
    sumvar = np.zeros(maxdist)
    N = np.zeros(maxdist, dtype=np.int32)
    # sample data if it's too big
    if resample > 1:
        sample_size = int(resample)
    elif resample != 0:
        sample_size = int(resample*s)
    else:
        sample_size = s
    sample = np.random.choice(np.arange(s), sample_size)
    for i in range(0, sample_size//2):
        for j in range(sample_size//2, sample_size):
            m = sample[i]
            n = sample[j]
            xi, xj, yi, yj = m%w, n%w, m//w, n//w
            distr = np.sqrt((xj-xi)**2+(yj-yi)**2)
            dist = int(np.round(distr))
            var = (z[n] - z[m])**2
            sumvar[dist] += var
            N[dist] += 1
    # here sumvar(dist) becomes meanvar(dist)
    for dist in range(maxdist):
        if N[dist] != 0:
            sumvar[dist] /= N[dist]
    return(sumvar)


def validate_semivariogram_index(semivariogram, x=None):
    if x is None:
        x = np.arange(1, semivariogram.size+1)
    index = np.logical_and(x>0, semivariogram>0)
    return(index)


def valid_semivariogram2d(img, resample=1/100):
    semivariogram = semivariogram(img, resample)
    x = np.arange(1, semivariogram.size+1)
    valid_index = validate_semivariogram_index(semivariogram, x=x)
    x = x[valid_index]
    semivariogram = semivariogram[valid_index]
    return(x,semivariogram)


def semivariogram_coefs(semivariogram=None, x=None, semivariogram_log=None, x_log=None):
    index = None
    if semivariogram is not None:
        if x is None:
            x = np.arange(1,semivariogram.size+1)
        index = validate_semivariogram_index(semivariogram, x)
        x, semivariogram  = x[index],  semivariogram[index]
        x_log, semivariogram_log = np.log(x), np.log(semivariogram)
    else:
        assert semivariogram_log is not None and x_log is not None
    coefs = np.polyfit(x_log, semivariogram_log, deg=1)
    return(coefs)


def semivariogram_vars_from_coefs(coefs):
    H = coefs[0]
    l = np.exp(coefs[1]/(2*(1-H)))
    D = 3-H
    return(H,l,D)


def semivariogram_frac_dim_from_coefs(coefs):
    return(3-coefs[0])


def plot_semivariogram(img, label=None, color=None, ax=None, s=1/100):
    semivariogram = semivariogram2d(img, s)
    x = np.arange(1,semivariogram.size+1)
    idx = validate_semivariogram_index(semivariogram, x)
    x, semivariogram  = x[idx],  semivariogram[idx]
    x_log, semivariogram_log = np.log(x), np.log(semivariogram)
    coefs = semivariogram_coefs(x_log=x_log, semivariogram_log=semivariogram_log)
    H, l, D = semivariogram_vars_from_coefs(coefs)
    poly = np.poly1d(coefs)
    pt, = plt.plot(x, semivariogram, label=label)
    pe, = plt.plot(np.exp(x_log), np.exp(poly(x_log)),
                   color=color,
                   ls='--',
                   label=f"D={D:.2f}, H={H:.2f}, l={l:.2f}",)
    ax.set_yscale('log')
    ax.set_xscale('log')
    return(pt,pe)


def fractal_dimension(img, resample=1/100):
    '''
    Calculates fractal dimension of 2D data based on semivariogram curve.
    :param img: 2D data array.
    :param resample: How many image pixel samples to use from before calculating
        semivariogram.
    :type img: 2D numpy.ndarray object.
    :type resample: absolute integer or sample percentage float.
    :return: fractal dimension.
    :rtype: float
    '''
    semivariogram = semivariogram2d(img, resample)
    coefs = semivariogram_coefs(semivariogram)
    D = semivariogram_frac_dim_from_coefs(coefs)
    return(D)


def files_semivariogram_plot(filepaths):
    fig, ax = plt.subplots()

    for i, filepath in enumerate(filepaths):
        filepath = expanduser(filepath)
        name = basename(splitext(filepath)[0])
        img = np.mean(np.array(Image.open(filepath)), axis=-1)
        img = ndimage.gaussian_filter(img, sigma=1)
        plot_semivariogram(img, label=name, ax=ax, color='C'+str(i))
    fig.legend()
    plt.show()


def files_fractal_dimension(filepaths):
    fractal_dimensions = []
    for i, filepath in enumerate(filepaths):
        filepath = expanduser(filepath)
        name = basename(splitext(filepath)[0])
        img = np.mean(np.array(Image.open(filepath)), axis=-1)
        img = ndimage.gaussian_filter(img, sigma=1)
        fractal_dimensions.append(fractal_dimension(img, 1/100))
    return(fractal_dimensions)


if __name__ == '__main__':
    #folder='/home/marcosrdac/Dropbox/projects/fractal_dimension/semivariogram_programming/test'
    folder='pics'
    filepaths = [ join(folder, i) for i in listdir(folder) ]
    filepaths.sort()
    #filepaths = ['~/Dropbox/pictures/wallpapers/favorites/boats_happy_sunrise.jpg',]
    semivariogram_Ds = files_fractal_dimension(filepaths)
    print(semivariogram_Ds)
