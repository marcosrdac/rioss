from datetime import datetime
from os import environ, listdir, mkdir
from os.path import expanduser, basename,  exists, isfile, isdir, relpath, dirname, join, splitext
from routines.functions import discarray, unsigned_span
import numpy as np
import netCDF4 as nc
import cv2 as cv
import matplotlib.pyplot as plt


def segmentate(img, ret_aux=False):
    img8 = unsigned_span(img)
    blur = cv.GaussianBlur(img8, (7, 7), 0)

    otsu = cv.threshold(blur, 0, 1,
                        cv.THRESH_BINARY+cv.THRESH_OTSU)[1]
    mean = cv.adaptiveThreshold(blur, 1,
                                cv.ADAPTIVE_THRESH_MEAN_C,
                                cv.THRESH_BINARY,
                                601, 4)
    total = otsu + mean
    total_blur = cv.medianBlur(total, 13)
    segmented = np.where(total_blur >= 2, 1, 0)

    if ret_aux:
        return blur, otsu, mean, total, total_blur, segmented
    else:
        return segmented


if __name__ == '__main__':
    ncfiles = True

    band_choices = [
        'Sigma0_IW1_VV_db',
        'Sigma0_IW2_VV_db',
        'Intensity_IW2_VV_db',
        'Sigma0_VV_db',
        'Sigma0_db',
    ]

    IN_DA = "/mnt/hdd/home/tmp/los/data/classification_blocks"
    IN_NC = "/mnt/hdd/home/tmp/los/data/original"
    TODAY = datetime.today().strftime('%Y%m%d')
    OUT = f'/mnt/hdd/home/tmp/los/data/maps/{TODAY}_seg'
    if not exists(OUT):
        mkdir(OUT)

    if ncfiles:
        ls = [join(IN_NC, f) for f in listdir(IN_NC) if f.endswith('.nc')]
    else:
        ls = [join(IN_DA, f) for f in listdir(IN_DA)]

    cmap='Greys_r'
    cmap='gray'

    for f in ls:
        name = splitext(basename(f))[0]

        stay = True
        if ncfiles:
            ncd = nc.Dataset(f)
            for band in band_choices:
                if band in ncd.variables:
                    img = np.asarray(ncd.variables[band])
                    # ncd.close()
                    break
            stay = False
        else:
            img = discarray(f)
        if not stay: continue

        blur, otsu, mean, total, total_blur, seg = segmentate(img, True)
        
        fig, axes = plt.subplots(3,2, figsize=(7,7))
        ax = axes[0,0]
        ax.set_title(r'$\sigma_0$')
        ax.imshow(img, vmin=-50, vmax=20, cmap=cmap)
        ax = axes[1,0]
        ax.set_title(r'Local thresholding')
        ax.imshow(mean, cmap=cmap)
        ax = axes[2,0]
        ax.set_title(r'Otsu thresholding')
        ax.imshow(otsu, cmap=cmap)
        ax = axes[0,1]
        ax.set_title(r'Summed outputs')
        ax.imshow(total, cmap=cmap)
        ax = axes[1,1]
        ax.set_title(r'Median filtered mask')
        ax.imshow(total_blur, cmap=cmap)
        ax = axes[2,1]
        ax.set_title(r'Resulting mask')
        ax.imshow(seg, cmap=cmap)
        
        for ax in axes.flat:
            ax.set_yticks([])
            ax.set_xticks([])

        fig.tight_layout()

        plt.savefig(join(OUT, name + '.png'))
        plt.show()
        plt.close(fig)
