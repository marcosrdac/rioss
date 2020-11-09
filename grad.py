import matplotlib.pyplot as plt
from routines.functions import discarray, unsigned_span, grad, grad_max, grad_mean, grad_median, lap, lap_max
from scipy.ndimage import gaussian_filter, laplace
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

    # print(f"{np.mean(grad)}\t{np.median(grad)}\t{np.max(grad)}")
    print(lap_max(img))
    # print(grad.shape)

    break

    # plt.subplot(221)
    # plt.imshow(img)
    # plt.colorbar()
    # plt.subplot(222)
    # plt.imshow(grad(img))
    # plt.colorbar()
    # plt.subplot(224)
    # plt.imshow(lap(img))
    # plt.colorbar()
    # plt.show()
