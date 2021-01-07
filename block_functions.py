import numpy as np
import scipy.stats as stats
from routines.functions import *
import routines.psd as psd
import routines.boxcounting as boxcounting
import routines.shannon as shannon


def mean(img, seg, glcm, segglcm):
    return np.mean(img)

def std(img, seg, glcm, segglcm):
    return np.std(img)


def skew(img, seg, glcm, segglcm):
    return stats.skew(img.ravel())


def kurt(img, seg, glcm, segglcm):
    return stats.kurtosis(img.ravel())


def fgmean(img, seg, glcm, segglcm):
    return apply_over_masked(img, seg, np.mean)


def bgmean(img, seg, glcm, segglcm):
    return apply_over_masked(img, ~seg, np.mean)


def fgstd(img, seg, glcm, segglcm):
    return apply_over_masked(img, seg, np.std)

def bgstd(img, seg, glcm, segglcm):
    return apply_over_masked(img, ~seg, np.std)

def fgobgstd(img, seg, glcm, segglcm):
    return fgstd(img, seg, glcm, segglcm)/bgstd(img, seg, glcm, segglcm)

def bgskew(img, seg, glcm, segglcm):
    return apply_over_masked(img, ~seg, stats.skew)

def fgrelarea(img, seg, glcm, segglcm):
    return area(seg)/np.multiply(*img.shape)


def fgrelper(img, seg, glcm, segglcm):
    return perimeter(seg)/np.multiply(*img.shape)


def fgperoarea(img, seg, glcm, segglcm):
    return perimeter(seg)/area(seg)


def psdfd(img, seg, glcm, segglcm):
    return psd.fractal_dimension(img)


def bcfd(img, seg, glcm, segglcm):
    return boxcounting.fractal_dimension(seg)


def bclac(img, seg, glcm, segglcm):
    return lacunarity(img)


def segbclac(img, seg, glcm, segglcm):
    return lacunarity(seg)


def gradmean(img, seg, glcm, segglcm):
    return grad_mean(img)

def gradmedian(img, seg, glcm, segglcm):
    return grad_median(img)

def entropy(img, seg, glcm, segglcm):
    return shannon.norm_entropy(img)

def segentropy(img, seg, glcm, segglcm):
    return shannon.norm_entropy(seg)

def _complexity(img, seg, glcm, segglcm):
    return complexity(seg)

def _spreading(img, seg, glcm, segglcm):
    return spreading(seg)

def gradmeanomedian(img, seg, glcm, segglcm):
    return gradmean(img, seg, glcm, segglcm)/gradmedian(img, seg, glcm, segglcm)

def segcorr(img, seg, glcm, segglcm):
    return get_correlation(segglcm)

def diss(img, seg, glcm, segglcm):
    return get_dissimilarity(glcm)

def ener(img, seg, glcm, segglcm):
    return get_energy(glcm)

def segener(img, seg, glcm, segglcm):
    return get_energy(segglcm)

def cont(img, seg, glcm, segglcm):
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
                   'gradmean': gradmean,
                   'gradmedian': gradmedian,
                   'gradmeanomedian': gradmeanomedian,
}
