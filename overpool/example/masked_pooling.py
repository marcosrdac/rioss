#!/usr/bin/env python3
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from overpool import overlapping_pool


# Three layered image example

# setting parameters
## image color vector average as img
img = np.median(np.asarray(Image.open('rgb.jpg')), axis=2)
whs = 128
def pool_func(img, window, mask):
    masked = np.ma.masked_array(img,
                    mask=mask[window[0][0]:window[0][1],
                              window[1][0]:window[1][1]])
    try:    result = np.nanmean(masked)
    except: result = np.mean(masked)
    return(result)


#defining mask
mask = np.ones_like(img)
#mask[:,::500] = 0
mask[:,3840//2:] = 0
masked = np.ma.masked_array(img, mask=mask)


# pooling
img_pool = overlapping_pool(img, whs, pool_func,
        give_window=True, pool_func_kw={'mask': mask})

# showing results
fig, axes = plt.subplots(3,1, figsize=(3,6.5))
axes[0].set_title('Original Image')
axes[0].imshow(img)
axes[1].set_title('Mask used')
axes[1].imshow(mask)
axes[2].set_title('Pooling layer')
axes[2].imshow(img_pool)
fig.subplots_adjust(hspace=.5)
plt.savefig('masked_pool.png')
plt.show()
