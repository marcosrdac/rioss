#!/usr/bin/env python3
import numpy as np
import netCDF4 as nc
from PIL import Image
import matplotlib.pyplot as plt
from overpool import overlapping_pool


# the resources for this example are not in this repository


img_fp = '''\
/home/marcosrdac/projects/oil_spill/netcdf/S1A_IW_SLC__1SDV_20181008T052755_20181008T052822_024040_02A081_3524_deb_Orb_ML_msk.nc\
'''
img = nc.Dataset(img_fp)['Intensity_VV_db']
print(img.shape)

whs = 512
def pool_func(x):
    try:
        val = np.nanvar(x)
    except:
        val = np.var(x)
    return(val)
img_pool = overlapping_pool(img, whs, pool_func)

fig, axes = plt.subplots(2,1)
axes[0].set_title('Original Image')
axes[0].imshow(img)
axes[1].set_title('Pooling layer')
axes[1].imshow(img_pool)
fig.subplots_adjust(hspace=.4)
plt.savefig('netcdf_pool.png')
plt.show()
