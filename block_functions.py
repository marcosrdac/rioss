import numpy as np
import scipy.stats as stats
from routines.functions import apply_over_masked, perimeter, area, lacunarity, \
        get_glcm, get_homogeneity, get_dissimilarity, get_correlation
import routines.psd as psd
import routines.boxcounting as boxcounting


def std(img, seg, glcm, segglcm):
    return np.nanstd(img)


def skew(img, seg, glcm, segglcm):
    return stats.skew(img.flatten())


def kurt(img, seg, glcm, segglcm):
    return stats.kurtosis(img.flatten())


def fgmean(img, seg, glcm, segglcm):
    return apply_over_masked(img, seg, np.mean)


def bgmean(img, seg, glcm, segglcm):
    return apply_over_masked(img, ~seg, np.mean)


def fgstd(img, seg, glcm, segglcm):
    return apply_over_masked(img, seg, np.nanstd)


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
    return lacunarity(seg)


def segglcmcorr(img, seg, glcm, segglcm):
    return get_correlation(segglcm)


# def glcmdiss(img, seg, glcm, segglcm):
    # return get_glcmdiss(glcm)




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
}
