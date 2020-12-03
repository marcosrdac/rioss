import numpy as np
import scipy.stats as stats
from routines.functions import apply_over_masked, perimeter, area, lacunarity, \
        get_glcm, get_homogeneity, get_dissimilarity, get_correlation
from routines.shannon import entropy
import routines.psd as psd
import routines.boxcounting as boxcounting


def std(img=None, seg=None, glcm=None, segglcm=None, grad=None):
    return np.std(img)


def skew(img=None, seg=None, glcm=None, segglcm=None, grad=None):
    return stats.skew(img.flatten())


def kurt(img=None, seg=None, glcm=None, segglcm=None, grad=None):
    return stats.kurtosis(img.flatten())


def fgmean(img=None, seg=None, glcm=None, segglcm=None, grad=None):
    return apply_over_masked(img, seg, np.mean)


def bgmean(img=None, seg=None, glcm=None, segglcm=None, grad=None):
    return apply_over_masked(img, ~seg, np.mean)


def fgstd(img=None, seg=None, glcm=None, segglcm=None, grad=None):
    return apply_over_masked(img, seg, np.std)


def fgrelarea(img=None, seg=None, glcm=None, segglcm=None, grad=None):
    return area(seg)/np.multiply(*img.shape)


def fgrelper(img=None, seg=None, glcm=None, segglcm=None, grad=None):
    return perimeter(seg)/np.multiply(*img.shape)


def fgperoarea(img=None, seg=None, glcm=None, segglcm=None, grad=None):
    return perimeter(seg)/area(seg)


def psdfd(img=None, seg=None, glcm=None, segglcm=None, grad=None):
    return psd.fractal_dimension(img)


def bcfd(img=None, seg=None, glcm=None, segglcm=None, grad=None):
    return boxcounting.fractal_dimension(seg)


def bclac(img=None, seg=None, glcm=None, segglcm=None, grad=None):
    return lacunarity(seg)


def segglcmcorr(img=None, seg=None, glcm=None, segglcm=None, grad=None):
    return get_correlation(segglcm)

def _entropy(img=None, seg=None, glcm=None, segglcm=None, grad=None):
    return entropy(img)


def gradmax(img=None, seg=None, glcm=None, segglcm=None, grad=None):
    return np.max(grad)

def gradmedian(img=None, seg=None, glcm=None, segglcm=None, grad=None):
    return np.median(grad)

def gradmean(img=None, seg=None, glcm=None, segglcm=None, grad=None):
    return np.mean(grad)

def gradmeanomedian(img=None, seg=None, glcm=None, segglcm=None, grad=None):
    return np.mean(grad)/np.median(grad)


BLOCK_FUNCTIONS = {'std': std,
                   'skew': skew,
                   'kurt': kurt,
                   'fgmean': fgmean,
                   'bgmean': bgmean,
                   'fgstd': fgstd,
                   'fgrelarea': fgrelarea,
                   'fgrelper': fgrelper,
                   'fgperoarea': fgperoarea,
                   'psdfd': psdfd,
                   'bcfd': bcfd,
                   'bclac': bclac,
                   'segglcmcorr': segglcmcorr,
                   'entropy': _entropy,
                   'gradmax': gradmax,
                   'gradmean': gradmean,
                   'gradmeanomedian': gradmeanomedian,
}
