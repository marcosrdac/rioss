import numpy as np
import scipy as scp
import scipy.signal as scpsig
from numba import jit, prange, stencil
from PIL import Image
import matplotlib.pyplot as plt


@jit(parallel=True, nopython=True, boundscheck=False,)
def pad(img, pad_radii, val=0):
    pady = pad_radii[0]
    padx = pad_radii[1]
    padded_img = np.empty((img.shape[0]+2*pady,
                           img.shape[1]+2*padx))
    padded_img[:pady, :] = val
    padded_img[-pady:, :] = val
    padded_img[pady:-pady, :padx] = val
    padded_img[pady:-pady, -padx:] = val
    padded_img[pady:-pady, padx:-padx] = img
    return(padded_img)


def get_convolver(kernel):
    wy = kernel.shape[0]
    wx = kernel.shape[1]
    wry = wy//2
    wrx = wx//2
    kernel_r = kernel[::-1, ::-1].copy()

    @stencil(neighborhood=((-wry, wry), (-wrx, wrx)))
    def stencil_kernel(arr):
        result = 0.
        window = arr[-wry:wry+1, -wrx:wrx+1, ]
        for i in prange(wy):
            for j in prange(wx):
                result += window[i, j] * kernel_r[i, j]
        return(result)

    @jit(parallel=True, nopython=True, cache=True)
    def filter(arr):
        _arr = pad(arr, (wry, wrx))
        _convolved = stencil_kernel(_arr)
        convolved = _convolved[wry:_arr.shape[0]-wry,
                               wrx:_arr.shape[1]-wrx]
        return(convolved)

    return(filter)


def get_mwa(wr):
    '''
    Returns a function of an array that returns a square moving window average 
    over it.
    '''
    ws = 2*wr+1
    kernel = np.full((ws, ws), ws**-2)
    mwa = get_convolver(kernel)
    return(mwa)


def get_mrwa(r_o, r_i=0):
    '''
    Returns a function of an array that returns a moving ring window average
    over it.
    '''
    ring = np.zeros((2*r_o+1, 2*r_o+1))

    for z in range(ring.shape[0]):
        for x in range(ring.shape[1]):
            dist = np.linalg.norm([r_o-z, r_o-x])
            if dist <= r_o+.3 and dist >= r_i+.3:
                ring[z, x] = 1
    ring /= np.sum(ring)

    mrwa = get_convolver(ring)
    return(mrwa)


@jit(parallel=True, nopython=True, boundscheck=False,)
def mwsd(img, wr=1):
    '''
    Calculates the moving window standard deviation of an array (img) padded 
    copy.
    '''
    stdevimg = np.empty(img.shape)  # change to empty
    img_pad = pad(img, (wr, wr))

    ws = 1+2*wr
    n2 = ws**2

    for y in prange(0, stdevimg.shape[0]):
        x = 0
        # first iteration
        subimg = img_pad[y:y+ws,
                         x:x+ws]

        oldvalssum = subimg[:, 0].sum()        # used in next iteration
        totalsum = subimg[:, 1:].sum() + oldvalssum

        oldsqvalssum = (subimg[:, 0]**2).sum()  # used in next iteration
        sqvalssum = (subimg[:, 1:]**2).sum() + oldsqvalssum

        stdevimg[y, x] = np.sqrt((n2*sqvalssum-totalsum**2) / n2**2)

        # next iterations
        for x in range(1, stdevimg.shape[1]):
            subimg = img_pad[y:y+ws,
                             x:x+ws]
            newvalssum = subimg[:, ws-1].sum()
            totalsum += newvalssum - oldvalssum
            newsqvalssum = (subimg[:, ws-1]**2).sum()
            sqvalssum += newsqvalssum - oldsqvalssum
            stdevimg[y, x] = np.sqrt((n2*sqvalssum-totalsum**2) / n2**2)

            oldvalssum = subimg[:, 0].sum()        # used in next iteration
            oldsqvalssum = (subimg[:, 0]**2).sum()  # used in next iteration
    return(stdevimg)


fn = "little_test_image.png"
#fn = "/home/marcosrdac/Dropbox/pictures/wallpapers/favorites/arara_azul_brazil_total_gc.jpg"


arr = np.mean(np.array(Image.open(fn)), 2)
#arr = np.where(arr>127, 1., 0.)


kernel = np.array([[1,   4, 1],
                   [4, -20, 4],
                   [1,   4, 1],])
filt = get_convolver(kernel)
#filt = get_mwa(7)
#filt = get_mrwa(4,3)
#filt = mwsd


convolved_mine = filt(arr)
convolved_scp = scpsig.convolve(arr, kernel, mode='same')

fig, axes = plt.subplots(2, 2)

axes[0, 0].set_title("Array")
axes[0, 0].imshow(arr)
axes[0, 1].set_title("Kernel")
axes[0, 1].imshow(kernel)
axes[1, 0].set_title("Scipy: Kernel∗Array")
axes[1, 0].imshow(convolved_scp)
axes[1, 1].set_title("Numba parallel: Kernel∗Array")
axes[1, 1].imshow(convolved_mine)
fig.tight_layout()
plt.show()
