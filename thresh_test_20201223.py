import matplotlib.pyplot as plt
from routines.functions import discarray, unsigned_span
import cv2 as cv
import numpy as np
from os.path import join, basename
from os import listdir, environ
# environ["CUDA_VISIBLE_DEVICES"] = "-1"
# from segmentation import segmentate

folder = "/mnt/hdd/home/tmp/los/data/classification_blocks"
ls = [join(folder, f) for f in listdir(folder)]

for f in ls:
    # f = ls[5]

    img = discarray(f)
    _img = unsigned_span(img)

    blur = cv.GaussianBlur(_img, (7, 7), 0)
    otsu = cv.threshold(blur, False, True,
                        cv.THRESH_BINARY+cv.THRESH_OTSU)[1]

    mean = cv.adaptiveThreshold(blur, 1,
                                cv.ADAPTIVE_THRESH_MEAN_C,
                                cv.THRESH_BINARY,
                                451, 4)

    total = otsu + mean

    total_blur = cv.medianBlur(total, 13)

    final = np.where(total_blur >= 2, 1, 0)

    plt.figure(figsize=(7,7))
    plt.subplot(221)
    # plt.title(r'$\sigma_0$')
    # plt.imshow(img, vmin=-50, vmax=20, cmap='Greys_r')
    plt.title(r'mean')
    plt.imshow(mean)
    plt.subplot(222)
    plt.title(r'adaptativos somados')
    plt.imshow(total)
    plt.subplot(224)
    plt.title(r'adaptativo resultante')
    plt.imshow(final)
    plt.subplot(223)
    plt.title(r'otsu')
    plt.imshow(otsu)
    # plt.savefig(f"/mnt/hdd/home/tmp/los/data/maps/20201110_threshblockpics/{basename(f)}.png")
    plt.show()
