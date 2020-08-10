from os import listdir
from os.path import join
import numpy as np
from numba import jit, prange, stencil
import scipy as scp
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from scipy.fftpack import fftn
from scipy.ndimage.filters import laplace, gaussian_filter
from scipy.signal import convolve2d
from scipy.spatial import KDTree


def discarray(filename, mode='r', dtype=float, shape=None):
    file_mode = f'{mode[0]}b{mode[1:]}'
    if not isinstance(shape, tuple):
        shape = (shape,)
    with open(filename, file_mode) as io:
        if 'w' in mode:
            ndims_shape = np.array((len(shape), *shape), dtype=np.int64)
            ndims_shape.tofile(io)
        if 'r' in mode:
            ndims = np.fromfile(io, dtype=np.int64, count=1)[0]
            shape = tuple(np.fromfile(io, dtype=np.int64, count=ndims))
        offset = io.tell()
        arr = np.memmap(io, dtype=dtype, mode=mode, offset=offset, shape=shape)
        return(arr)


def listifext(folder, exts=None, fullpath=False):
    filepaths = listdir(folder)
    if exts:
        if isinstance(exts, str):
            exts = [exts]
        filepaths = [f for f in filepaths
                     for ext in exts
                     if f.endswith(f'.{ext}')]
        if fullpath:
            filepaths = [join(folder, f) for f in filepaths]
    return(filepaths)


def first_channel(img):
    if len(img.shape) == 3:
        channel = np.squeeze(np.split(img, img.shape[-1], -1), axis=-1)
    elif len(img.shape) == 2:
        channel = img
    else:
        raise Exception(f"Array has {len(img.shape)} dimensions!.")
    return(channel)


def sample_min_dist(bool_arr, dist, sample=.1, skip='auto'):
    if skip == 'auto':
        skip = dist//2
    pts = np.stack(np.where(bool_arr), axis=1)[::skip].copy()
    N = pts.shape[0]
    accepted = np.ones(N, dtype=bool)
    if N != 0:
        kdt = KDTree(pts)
        for p in range(N):
            if accepted[p]:
                pt = pts[p, :]
                near_pts = kdt.query_ball_point(pt, dist)
                for q in near_pts:
                    if q != p:
                        accepted[q] = False
    sampled_pts = pts[accepted]
    return(sampled_pts)


@jit(parallel=True, nopython=True, boundscheck=False, cache=True)
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


@jit(parallel=True, nopython=True, boundscheck=False, cache=True)
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


@jit(parallel=True, nopython=True, boundscheck=False, cache=True)
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


@jit(nopython=True, cache=True)
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


def get_mcwa(r):
    '''
    Returns a function of an array that returns a moving circular window average
    over it.
    '''
    mcwa = get_mrwa(r, 0)
    return(mcwa)


@jit(parallel=True, nopython=True, boundscheck=False, cache=True)
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


def fn2ar(filename):
    img = Image.open(filename)
    img = np.array(img)
    if len(img.shape) == 3:
        img = rgb2gray(img)
        return(img)
    elif len(img.shape) != 1:
        raise('Number of channels unhandled.')


def plot(grid, fig=None, ax=None, contourf=True, four=None, phase=None, title=None,
         vmin=None, vmax=None,
         log=False,
         figsize=None, dpi=None):
    if not fig and not ax:
        fig, ax = plt.subplots(figsize=None, dpi=None)

    ax.set_title(title)
    if four:
        dfour = np.fft.fft2(grid)
        dfour = np.fft.fftshift(dfour)
        if phase:
            grid = np.angle(dfour)
        else:
            grid = np.abs(dfour)
        y = np.arange(-grid.shape[0]//2,
                      -grid.shape[0]//2 + grid.shape[0])
        x = np.arange(-grid.shape[1]//2,
                      -grid.shape[1]//2 + grid.shape[1])
    else:
        y = np.arange(grid.shape[0])
        x = np.arange(grid.shape[1])
    x, y = np.meshgrid(x, y)

    if log:
        norm = colors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
    if contourf:
        ax.invert_yaxis()
        contourf = ax.contourf(x, y, grid, norm=norm)
        cbar = fig.colorbar(contourf, ax=ax)
    else:
        if log:
            sns.heatmap(grid, ax=ax, vmin=vmin, vmax=vmax, norm=norm,
                        cbar=False)
        else:
            sns.heatmap(grid, ax=ax, vmin=vmin, vmax=vmax)


def plot23(grid, grid_filt, sources=None, prop='$\Delta T$',
           title='', figname=None, scale=True, log1=True, log2=True, dpi=None):
    # ,constrained_layout=True)
    fig, axes = plt.subplots(2, 3, figsize=(11, 7), dpi=dpi)

    if sources:
        sources = np.array(sources)
        sourcex = sources[:, 0]
        sourcey = sources[:, 1]
    else:
        sourcex = np.array([])
        sourcey = np.array([])

    plot(grid, fig, axes[0, 0], title=prop+'\n(domínio do espaço)',
         vmin=grid.min(), vmax=grid.max())
    axes[0, 0].scatter(sourcex, sourcey, c='white', s=3)
    plot(grid, fig, axes[0, 1], title='Amp. do '+prop +
         ' transformado\n(domínio do n. de onda)',
         four=True, log=log1)
    plot(grid, fig, axes[0, 2], title='Fase do '+prop +
         ' transformado\n(domínio do n. de onda)',
         four=True, phase=True)
    if scale:
        plot(grid_filt, fig, axes[1, 0],
             title=prop+' filtrado\n(domínio do espaço)')
    else:
        plot(grid_filt, fig, axes[1, 0],
             title=prop+' filtrado\n(domínio do espaço)',
             vmin=grid.min(), vmax=grid.max())
    axes[1, 0].scatter(sourcex, sourcey, c='white', s=3)
    plot(grid_filt, fig, axes[1, 1], title='Amp. do '+prop +
         ' filtrado\n(domínio do n. de onda)', four=True, log=log2)
    plot(grid_filt, fig, axes[1, 2], title='Fase do '+prop +
         ' filtrado\n(domínio do n. de onda)',
         four=True, phase=True)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    if figname:
        fig.savefig(figname+'.png')
    plt.show()
    plt.close(fig)
