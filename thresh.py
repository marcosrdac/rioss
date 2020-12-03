import matplotlib.pyplot as plt
from routines.functions import discarray, unsigned_span
import cv2 as cv
import numpy as np
from os.path import join, basename
from os import listdir, environ
environ["CUDA_VISIBLE_DEVICES"] = "-1"
from segmentation import segmentate
# from segmentation import segmentate

folder = "/mnt/hdd/home/tmp/los/data/classification_blocks"
ls = [join(folder, f) for f in listdir(folder)]

for f in ls:
    # f = ls[5]

    img = discarray(f)
    _img = unsigned_span(img)

    blur = cv.GaussianBlur(_img, (7, 7), 0)
    thresh = cv.threshold(blur, False, True,
                          cv.THRESH_BINARY+cv.THRESH_OTSU)[1]

    thresh += cv.adaptiveThreshold(blur, 1,
                                   cv.ADAPTIVE_THRESH_MEAN_C,
                                   cv.THRESH_BINARY,
                                   401, 4)

    blur = cv.medianBlur(thresh, 13)
    print(blur.min(), blur.max())
    # thresh = blur >= 2
    _thresh = 1-np.where(blur >= 2, 0, 1)

    segmented = segmentate(img)

    plt.figure(figsize=(7,7))
    plt.subplot(221)
    plt.title(r'$\sigma_0$')
    plt.imshow(img, vmin=-50, vmax=20, cmap='Greys_r')
    plt.subplot(222)
    plt.title(r'adaptativos somados')
    plt.imshow(thresh)
    plt.subplot(224)
    plt.title(r'adaptativo resultante')
    plt.imshow(_thresh)
    plt.subplot(223)
    plt.title(r'rede neural')
    plt.imshow(segmented)
    plt.savefig(f"/mnt/hdd/home/tmp/los/data/maps/20201110_threshblockpics/{basename(f)}.png")
    # plt.show()
