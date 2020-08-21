from os import listdir
from os.path import basename, splitext, join, realpath, expanduser
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from numba import jit, prange
from tqdm import tqdm
from PIL import Image
from .functions import timer


def fourier_transform(img):
    imgfour = np.fft.fft2(img)
    imgfour = np.fft.fftshift(imgfour)
    return(imgfour)


def radial_psd_2d(img):
    imgfour = fourier_transform(img)
    imgpsd2d = np.real(imgfour * np.conj(imgfour))
    return(imgpsd2d)


# Modified from
# https://medium.com/tangibit-studios/2d-spectrum-characterization-e288f255cc59
def radial_psd_1d(img=None, imgpsd2d=None):
    if imgpsd2d is None:
        imgpsd2d = radial_psd_2d(img)
    h  = imgpsd2d.shape[0]
    w  = imgpsd2d.shape[1]
    wc = w//2
    hc = h//2

    # create an array of integer radial distances from the center
    Y, X = np.ogrid[0:h, 0:w]
    r    = np.hypot(X - wc, Y - hc).astype(np.int)

    # SUM all psd2D pixels with label 'r' for 0<=r<=wc
    # NOTE: this will miss power contributions in 'corners' r>wc
    imgpsd1d = ndimage.sum(imgpsd2d, r, index=np.arange(0, wc))
    return(imgpsd1d)


def log_log_psd_coefs(img=None, psd1d=None, psd_log=None, x_log=None):
    if psd_log is None:
        if psd1d is None:
            psd1d = radial_psd_1d(img)
        psd_log = np.log(psd1d)
    if x_log is None:
        x = np.arange(1, psd1d.size+1)
        x_log = np.log(x)
    coefs = np.polyfit(x_log, psd_log, 1)
    return(coefs)

# DOI: 10.1109/IGARSS.1999.773452 · Source: IEEE Xplore
def fractal_dimension(img=None, coefs=None):
    if coefs is None:
        coefs = log_log_psd_coefs(img)
    beta = -coefs[0]
    H = (beta - 2)/2
    D = 3 - H
    return(D)


def plot_psd(img, label=None, color=None, ax=None, s=1/100):
    psd = radial_psd_1d(img)
    x = np.arange(1, psd.size+1)
    x_log, psd_log = np.log(x), np.log(psd)
    coefs = log_log_psd_coefs(x_log=x_log, psd_log=psd_log)
    poly = np.poly1d(coefs)
    W = -coefs[0]
    D = fractal_dimension(coefs=coefs)
    pt, = plt.plot(x, psd, label=label)
    pe, = plt.plot(x, np.exp(poly(x_log)),
                   color=color,
                   ls='--',
                   label=f"D={D:.2f}, W={W:.2f}")
    ax.set_yscale('log')
    ax.set_xscale('log')
    return(pt,pe)


def files_psd_plot(filepaths):
    fig, ax = plt.subplots()
    for i, filepath in enumerate(filepaths):
        filepath = expanduser(filepath)
        name = basename(splitext(filepath)[0])
        img = np.mean(np.array(Image.open(filepath)), axis=-1)
        plot_psd(img, ax=ax, color='C'+str(i), label=name)
    fig.legend()
    plt.show()

def files_fractal_dimension(filepaths):
    fractal_dimensions = []
    for i, filepath in enumerate(filepaths):
        filepath = expanduser(filepath)
        name = basename(splitext(filepath)[0])
        img = np.mean(np.array(Image.open(filepath)), axis=-1)
        fractal_dimensions.append(fractal_dimension(img))
    return(fractal_dimensions)


if __name__ == '__main__':
    #folder='/home/marcosrdac/Dropbox/projects/fractal_dimension/semivariogram_programming/test'
    folder='pics'
    filepaths = [ join(folder, i) for i in listdir(folder) ]
    filepaths.sort()
    #filepaths = ['~/Dropbox/pictures/wallpapers/favorites/boats_happy_sunrise.jpg',]
    psd_Ds = files_fractal_dimension(filepaths)
    print(psd_Ds)
