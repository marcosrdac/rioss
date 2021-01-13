import numpy as np
import scipy.stats as stats
from routines.functions import *
import routines.psd as psd
import routines.boxcounting as boxcounting
import routines.shannon as shannon


def mean(img=None, seg=None, glcm=None, segglcm=None, grad=None):
    return np.mean(img)

def std(img=None, seg=None, glcm=None, segglcm=None, grad=None):
    return np.std(img)


def skew(img=None, seg=None, glcm=None, segglcm=None, grad=None):
    return stats.skew(img.ravel())


def kurt(img=None, seg=None, glcm=None, segglcm=None, grad=None):
    return stats.kurtosis(img.ravel())


def fgmean(img=None, seg=None, glcm=None, segglcm=None, grad=None):
    return apply_over_masked(img, seg, np.mean)


def bgmean(img=None, seg=None, glcm=None, segglcm=None, grad=None):
    return apply_over_masked(img, ~seg, np.mean)


def fgstd(img=None, seg=None, glcm=None, segglcm=None, grad=None):
    return apply_over_masked(img, seg, np.std)


def bgstd(img=None, seg=None, glcm=None, segglcm=None, grad=None):
    return apply_over_masked(img, ~seg, np.std)


def fgobgstd(img=None, seg=None, glcm=None, segglcm=None, grad=None):
    return fgstd(img, seg, glcm, segglcm)/bgstd(img, seg, glcm, segglcm)


def bgskew(img=None, seg=None, glcm=None, segglcm=None, grad=None):
    return apply_over_masked(img, ~seg, stats.skew)


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
    return lacunarity(img)


def segbclac(img=None, seg=None, glcm=None, segglcm=None, grad=None):
    return lacunarity(seg)




def gradmax(img=None, seg=None, glcm=None, segglcm=None, grad=None):
    return np.max(grad)

def gradmedian(img=None, seg=None, glcm=None, segglcm=None, grad=None):
    return np.median(grad)

def gradmean(img=None, seg=None, glcm=None, segglcm=None, grad=None):
    return np.mean(grad)

def gradmeanomedian(img=None, seg=None, glcm=None, segglcm=None, grad=None):
    return np.mean(grad)/np.median(grad)



def entropy(img=None, seg=None, glcm=None, segglcm=None, grad=None):
    return shannon.norm_entropy(img)

def segentropy(img=None, seg=None, glcm=None, segglcm=None, grad=None):
    return shannon.norm_entropy(seg)

def _complexity(img=None, seg=None, glcm=None, segglcm=None, grad=None):
    return complexity(seg)

def _spreading(img=None, seg=None, glcm=None, segglcm=None, grad=None):
    return spreading(seg)

def segcorr(img=None, seg=None, glcm=None, segglcm=None, grad=None):
    return get_correlation(segglcm)

def diss(img=None, seg=None, glcm=None, segglcm=None, grad=None):
    return get_dissimilarity(glcm)

def ener(img=None, seg=None, glcm=None, segglcm=None, grad=None):
    return get_energy(glcm)

def segener(img=None, seg=None, glcm=None, segglcm=None, grad=None):
    return get_energy(segglcm)

def cont(img=None, seg=None, glcm=None, segglcm=None, grad=None):
    return get_contrast(glcm)


BLOCK_FUNCTIONS = {
                   'mean': mean,
                   'std': std,
                   'skew': skew,
                   'bgskew': bgskew,
                   'kurt': kurt,
                   'fgmean': fgmean,
                   'bgmean': bgmean,
                   'fgstd': fgstd,
                   'bgstd': bgstd,
                   'fgobgstd': fgobgstd,
                   'fgrelarea': fgrelarea,
                   'fgrelper': fgrelper,
                   'fgperoarea': fgperoarea,
                   'psdfd': psdfd,
                   'bcfd': bcfd,
                   'bclac': bclac,
                   'segcorr': segcorr,
                   'ener': ener,
                   'segener': segener,
                   'cont': cont,
                   'entropy': entropy,
                   'segentropy': segentropy,
                   'complex': _complexity,
                   'spreading': _spreading,
                   'gradmax': gradmax,
                   'gradmean': gradmean,
                   'gradmedian': gradmedian,
                   'gradmeanomedian': gradmeanomedian,
}
