import matplotlib.pyplot as plt
from routines.functions import discarray, unsigned_span
import cv2 as cv
import numpy as np
from os.path import join
from os import listdir, environ

def segmentate(img):
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
    # print(blur.min(), blur.max())  # 0 2
    thresh = np.where(blur >= 2, 1, 0)
    return thresh
