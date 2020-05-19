#!/usr/bin/env python3
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from overpool import overlapping_pool


# Three layered image example

# setting parameters
img = np.asarray(Image.open('rgb.jpg'))
whs = 64
pool_func = np.ptp

# getting shape of 2D output in order to allocate 3D array
img_pool0 = overlapping_pool(img[:,:,0], whs, pool_func)
img_pool = np.empty((img_pool0.shape[0], img_pool0.shape[1], 3))
img_pool[:,:,0] = img_pool0; del(img_pool0)
# calculating green and blue channels
img_pool[:,:,1] = overlapping_pool(img[:,:,1], whs, pool_func)
img_pool[:,:,2] = overlapping_pool(img[:,:,2], whs, pool_func)

# rescaling needed for RGB plotting
img_pool /= np.max(img_pool)

fig, axes = plt.subplots(2,1)
axes[0].set_title('Original Image')
axes[0].imshow(img)
axes[1].set_title('Pooling layer')
axes[1].imshow(img_pool)
fig.subplots_adjust(hspace=.4)
plt.savefig('rgb_pool.png')
plt.show()
