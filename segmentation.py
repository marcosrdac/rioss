import matplotlib.pyplot as plt
from routines.functions import discarray, unsigned_span
import cv2 as cv
import numpy as np
from os.path import join, basename
from os import listdir, environ


def segmentate(img, ret_aux=False):
    img8 = unsigned_span(img)
    blur = cv.GaussianBlur(img8, 7), 0)

    otsu = cv.threshold(blur, False, True,
                        cv.THRESH_BINARY+cv.THRESH_OTSU)[1]
    mean = cv.adaptiveThreshold(blur, 1,
                                cv.ADAPTIVE_THRESH_MEAN_C,
                                cv.THRESH_BINARY,
                                451, 4)
    total = otsu + mean
    total_blur = cv.medianBlur(total, 13)
    segmented = np.where(total_blur >= 2, 1, 0)

    if ret_aux:
        return blur, otsu, mean, total, total_blur, segmented
    else:
        return segmented


if __name__ == '__main__':
    folder = "/mnt/hdd/home/tmp/los/data/classification_blocks"
    ls = [join(folder, f) for f in listdir(folder)]

    cmap='Greys_r'
    cmap='gray'

    for f in ls:
        img = discarray(f)
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

        saving_folder = f"/mnt/hdd/home/tmp/los/data/maps/20210113_segmentation"
        plt.savefig(join(saving_folder, f"{basename(f)}.png"))
        # plt.show()
        plt.close(fig)
