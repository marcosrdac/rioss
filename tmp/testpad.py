import numpy as np
import scipy as scp
import scipy.signal as scpsig
from numba import jit, prange, stencil
from PIL import Image
import matplotlib.pyplot as plt


@jit(parallel=True, nopython=True, boundscheck=False,)
def non_parallel_pad_const(img, pad_radii, const=0):
    pady = pad_radii[0]
    padx = pad_radii[1]
    h = img.shape[0]
    w = img.shape[1]
    _img = np.empty((h+2*pady,  # change to zeros
                     w+2*padx))
    _h = _img.shape[0]
    _w = _img.shape[1]

    _img[:pady, :] = const
    _img[-pady:, :] = const
    _img[pady:-pady, :padx] = const
    _img[pady:-pady, -padx:] = const
    _img[pady:_h-pady,
               padx:_w-padx] = img
    return(_img)


@jit(parallel=True, nopython=True, boundscheck=False,)
def pad_const(img, pad_radii, const=0):
    pady = pad_radii[0]
    padx = pad_radii[1]
    h = img.shape[0]
    w = img.shape[1]
    _img = np.empty((h+2*pady,  # change to zeros
                     w+2*padx))
    _h = _img.shape[0]
    _w = _img.shape[1]

    # top left
    for y in prange(0, pady):
        for x in prange(0, padx):
            _img[y, x] = const
    # top
    for y in prange(0, pady):
        for x in prange(padx, _w-padx):
            _img[y, x] = const
    # top right
    for y in prange(0, pady):
        for x in prange(_w-padx, _w):
            _img[y, x] = const
    # left
    for y in prange(pady, _h-pady+1):
        for x in prange(0, padx):
            _img[y, x] = const
    # center
    _img[pady:-pady, padx:-padx] = img
    # right
    for y in prange(pady, _h-pady+1):
        for x in prange(_w-padx, _w):
            _img[y, x] = const
    # bottom left
    for y in prange(_h-pady, _h):
        for x in prange(0, padx):
            _img[y, x] = const
    # bottom
    for y in prange(_h-pady, _h):
        for x in prange(padx, _w-padx):
            _img[y, x] = const
    # bottom right
    for y in prange(_h-pady, _h):
        for x in prange(_w-padx, _w):
            _img[y, x] = const

    return(_img)

@jit(parallel=True, nopython=True, boundscheck=False,)
def pad_borders(img, pad_radii):
    pady = pad_radii[0]
    padx = pad_radii[1]
    h = img.shape[0]
    w = img.shape[1]
    _img = np.empty((h+2*pady,  # change to zeros
                     w+2*padx))
    _h = _img.shape[0]
    _w = _img.shape[1]

    # top left
    for y in prange(0, pady):
        for x in prange(0, padx):
            _img[y, x] = img[0, 0]
    # top
    for y in prange(0, pady):
        for x in prange(padx, _w-padx):
            _img[y, x] = img[0, x-padx]
    # top right
    for y in prange(0, pady):
        for x in prange(_w-padx, _w):
            _img[y, x] = img[0, -1]
    # bottom left
    for y in prange(_h-pady, _h):
        for x in prange(0, padx):
            _img[y, x] = img[-1, 0]
    # bottom
    for y in prange(_h-pady, _h):
        for x in prange(padx, _w-padx):
            _img[y, x] = img[-1, x-padx]
    # bottom right
    for y in prange(_h-pady, _h):
        for x in prange(_w-padx, _w):
            _img[y, x] = img[-1, -1]
    # left
    for y in prange(pady, _h-pady+1):
        for x in prange(0, padx):
            _img[y, x] = img[y-pady, 0]
    # center
    _img[pady:-pady, padx:-padx] = img
    # right
    for y in prange(pady, _h-pady+1):
        for x in prange(_w-padx, _w):
            _img[y, x] = img[y-pady, w-1]

    return(_img)


@jit(parallel=True, nopython=True, boundscheck=False,)
def pad_symmetric(img, pad_radii):
    # def pad_symmetric(img, pad_radii):
    pady = pad_radii[0]
    padx = pad_radii[1]
    h = img.shape[0]
    w = img.shape[1]
    _img = np.empty((h+2*pady,  # change to zeros
                     w+2*padx))
    _h = _img.shape[0]
    _w = _img.shape[1]

    # top left
    for y in prange(0, pady):
        for x in prange(0, padx):
            _img[y, x] = img[pady-y, padx-x]
    # top
    for y in prange(0, pady):
        for x in prange(padx, _w-padx):
            _img[y, x] = img[pady-y, x-padx]
    # top right
    for y in prange(0, pady):
        for x in prange(_w-padx, _w):
            _img[y, x] = img[pady-y, w-x]
    # left
    for y in prange(pady, _h-pady+1):
        for x in prange(0, padx):
            _img[y, x] = img[y-pady, padx-x]
    # center
    _img[pady:-pady, padx:-padx] = img
    # right
    for y in prange(pady, _h-pady+1):
        for x in prange(_w-padx, _w):
            _img[y, x] = img[y-pady, w-x]
    # bottom left
    for y in prange(_h-pady, _h):
        for x in prange(0, padx):
            _img[y, x] = img[h-y, padx-x]
    # bottom
    for y in prange(_h-pady, _h):
        for x in prange(padx, _w-padx):
            _img[y, x] = img[h-y, x-padx]
    # bottom right
    for y in prange(_h-pady, _h):
        for x in prange(_w-padx, _w):
            _img[y, x] = img[h-y, w-x]

    return(_img)


@jit(nopython=True)
def pad(arr, pad_radii, mode='const', const=0):
    if mode == 'const':
        _arr = pad_const(arr, pad_radii, const)
    elif mode == 'borders':
        _arr = pad_borders(arr, pad_radii)
    elif mode == 'symmetric':
        _arr = pad_symmetric(arr, pad_radii)
    return(_arr)



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

    @jit(parallel=True, nopython=True)
    def filter(arr):
        _arr = pad(arr, (wry, wrx), mode="symmetric")
        _convolved = stencil_kernel(_arr)
        convolved = _convolved[wry:_arr.shape[0]-wry,
                               wrx:_arr.shape[1]-wrx]
        return(convolved)  # CHANGE BACK TO convolved

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
    img_pad = pad(img, (wr, wr), mode='symmetric')

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


#fn = "little_test_image.png"
fn = "/home/marcosrdac/res/wal/favorites/beach_pastel.jpg"


arr = np.mean(np.array(Image.open(fn)), 2)
print(arr.shape)
# arr = np.where(arr>127, 1., 0.)


#kernel = np.array([[1,   4, 1],
#                   [4, -20, 4],
#                   [1,   4, 1],])
kernel = np.array([[0, 0, 0, 0, 0],
                   [0, 0, 1, 0, 0],
                   [0, 1, 1, 1, 0],
                   [0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 0], ])
filt = get_convolver(kernel)
# filt = get_mwa(7)
# filt = get_mrwa(4,3)
#filt = mwsd

convolved_mine = filt(arr)
convolved_scp = scpsig.convolve(arr, kernel, mode='same')

vmin = convolved_scp.min()
vmax = convolved_scp.max()

fig, axes = plt.subplots(2, 2)

axes[0, 0].set_title("Array")
axes[0, 0].imshow(arr)
axes[0, 1].set_title("Kernel")
axes[0, 1].imshow(kernel)
axes[1, 0].set_title("Scipy: Kernel∗Array")
axes[1, 0].imshow(convolved_scp, vmin=vmin, vmax=vmax)
axes[1, 1].set_title("Parallel w/ sym. pad: Kernel∗Array")
axes[1, 1].imshow(convolved_mine, vmin=vmin, vmax=vmax)
fig.tight_layout()
plt.show()
