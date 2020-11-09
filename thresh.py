import matplotlib.pyplot as plt
from routines.functions import discarray, unsigned_span
import cv2 as cv
import numpy as np
from os.path import join
from os import listdir, environ
environ["CUDA_VISIBLE_DEVICES"] = "-1"
# from segmentation import segmentate

folder = "/mnt/hdd/home/tmp/los/data/classification_blocks"
ls = [join(folder, f) for f in listdir(folder)]

for f in ls:
    # f = ls[5]

    img = discarray(f)
    # thresh = segmentate()
    _img = unsigned_span(img)

    blur = cv.GaussianBlur(_img, (7, 7), 0)
    thresh = cv.threshold(blur, False, True,
                          cv.THRESH_BINARY+cv.THRESH_OTSU)[1]

    # segmented = segmentate(img)

    thresh += cv.adaptiveThreshold(blur, 1,
                                   cv.ADAPTIVE_THRESH_MEAN_C,
                                   cv.THRESH_BINARY,
                                   601, 2)

    blur = cv.medianBlur(thresh, 11)
    print(blur.min(), blur.max())
    # thresh = blur >= 2
    thresh = np.where(blur >= 2, 1, 0)

    # thresh = cv.adaptiveThreshold(thresh, 2,
                                   # cv.ADAPTIVE_THRESH_MEAN_C,
                                   # cv.THRESH_BINARY,
                                   # 51, 2)


    plt.subplot(121)
    # plt.subplot(221)
    plt.imshow(img)
    plt.colorbar()
    plt.subplot(122)
    # plt.subplot(222)
    plt.imshow(thresh)
    plt.colorbar()
    # plt.subplot(224)
    # plt.imshow(segmented)
    # plt.colorbar()
    plt.show()
