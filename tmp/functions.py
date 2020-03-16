import numpy as np
import scipy as scp
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from scipy.fftpack import fftn
from scipy.ndimage.filters import laplace, gaussian_filter
from scipy.signal import convolve2d


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
        dfour  = np.fft.fft2(grid)
        dfour  = np.fft.fftshift(dfour)
        if phase:
            grid   = np.angle(dfour)
        else:
            grid   = np.abs(dfour)
        y = np.arange(-grid.shape[0]//2,
                      -grid.shape[0]//2 + grid.shape[0])
        x = np.arange(-grid.shape[1]//2,
                      -grid.shape[1]//2 + grid.shape[1])
    else:
        y = np.arange(grid.shape[0])
        x = np.arange(grid.shape[1])
    x,y = np.meshgrid(x,y)

    if log:
        norm = colors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
    if contourf:
        ax.invert_yaxis()
        contourf = ax.contourf(x,y,grid, norm=norm)
        cbar = fig.colorbar(contourf, ax=ax)
    else:
        if log:
            sns.heatmap(grid, ax=ax, vmin=vmin, vmax=vmax, norm=norm,
                    cbar=False)
        else:
            sns.heatmap(grid, ax=ax, vmin=vmin, vmax=vmax)

def plot23(grid, grid_filt, sources=None, prop='$\Delta T$',
        title='', figname=None, scale=True, log1=True, log2=True, dpi=None):
    fig, axes = plt.subplots(2,3, figsize=(11,7), dpi=dpi) #,constrained_layout=True)

    if sources:
        sources = np.array(sources)
        sourcex = sources[:,0]
        sourcey = sources[:,1]
    else:
        sourcex = np.array([])
        sourcey = np.array([])

    plot(grid, fig, axes[0,0], title=prop+'\n(domínio do espaço)',
            vmin=grid.min(), vmax=grid.max())
    axes[0,0].scatter(sourcex, sourcey, c='white', s=3)
    plot(grid, fig, axes[0,1], title='Amp. do '+prop+
                    ' transformado\n(domínio do n. de onda)',
                    four=True, log=log1)
    plot(grid, fig, axes[0,2], title='Fase do '+prop+
                    ' transformado\n(domínio do n. de onda)',
                    four=True, phase=True)
    if scale:
        plot(grid_filt, fig, axes[1,0],
                title=prop+' filtrado\n(domínio do espaço)')
    else:
        plot(grid_filt, fig, axes[1,0],
                title=prop+' filtrado\n(domínio do espaço)',
                vmin=grid.min(), vmax=grid.max())
    axes[1,0].scatter(sourcex, sourcey, c='white', s=3)
    plot(grid_filt, fig, axes[1,1], title='Amp. do '+prop+
                ' filtrado\n(domínio do n. de onda)', four=True, log=log2)
    plot(grid_filt, fig, axes[1,2], title='Fase do '+prop+
                ' filtrado\n(domínio do n. de onda)',
                four=True, phase=True)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    if figname:
        fig.savefig(figname+'.png')
    plt.show()
    plt.close(fig)

def fractal_dimension(Z, threshold=0.9):

    # Only for 2d image
    assert(len(Z.shape) == 2)

    # From https://github.com/rougier/numpy-100 (#87)
    def boxcount(Z, k):
        S = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                               np.arange(0, Z.shape[1], k), axis=1)

        # We count non-empty (0) and non-full boxes (k*k)
        return len(np.where((S > 0) & (S < k*k))[0])


    # Transform Z into a binary array
    Z = (Z < threshold)

    # Minimal dimension of image
    p = min(Z.shape)

    # Greatest power of 2 less than or equal to p
    n = 2**np.floor(np.log(p)/np.log(2))

    # Extract the exponent
    n = int(np.log(n)/np.log(2))

    # Build successive box sizes (from 2**n down to 2**1)
    sizes = 2**np.arange(n, 1, -1)

    # Actual box counting with decreasing size
    counts = []
    for size in sizes:
        counts.append(boxcount(Z, size))

    # Fit the successive log(sizes) with log (counts)
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]


from scipy import ndimage
#===================================================================
# Get PSD 1D (total radial power spectrum)
#===================================================================
def GetPSD1D(psd2D):
    h  = psd2D.shape[0]
    w  = psd2D.shape[1]
    wc = w//2
    hc = h//2

    # create an array of integer radial distances from the center
    Y, X = np.ogrid[0:h, 0:w]
    r    = np.hypot(X - wc, Y - hc).astype(np.int)

    # SUM all psd2D pixels with label 'r' for 0<=r<=wc
    # NOTE: this will miss power contributions in 'corners' r>wc
    psd1D = ndimage.sum(psd2D, r, index=np.arange(0, wc))

    return psd1D
#=============================================================================

def open_img(filename):
    img = np.array(Image.open(filename),
                            dtype=np.uint8)
    return(img)

def get_dfour(d):
    dfour = np.fft.fft2(d)
    dfour = np.fft.fftshift(dfour)
    return(dfour)

def get_rpsd(d):
    dfour = get_dfour(d)
    dfour_power = np.real(np.conj(dfour)*dfour)
    psd   = GetPSD1D(dfour_power)
    return(psd)

def llpsdd(img):
    psd = get_rpsd(img)
    a,_ = np.polyfit(np.arange(1,np.min(img.shape)//2+1),
                     np.log(psd),
                     1)
    return(a)


def rpsd(filename):
    return(get_rpsd(open_img(filename)))

def moving_mean(img, ws=3):
    return(
            convolve2d(img,
                       np.full((ws,ws), ws**-2),
                       mode='same', boundary='symm')
          )

def moving_circular_mean(img, r_o=5, r_i=0):
    circle=np.zeros((2*r_o+1, 2*r_o+1))

    for z in range(circle.shape[0]):
        for x in range(circle.shape[1]):
            dist=np.linalg.norm([r_o-z,r_o-x])
            if dist <= r_o+.3 and dist >= r_i+.3:
                circle[z,x] = 1
    circle /= np.sum(circle)

    print(circle.shape)

    return(
            convolve2d(img,
                       circle,
                       mode='same', boundary='symm')
          )

def std_convoluted(img, N):

    kernel = np.ones((2*N+1, 2*N+1))
    s  = convolve2d(img, kernel, mode="same")
    s2 = convolve2d(img**2, kernel, mode="same")
    imgshape = img.shape
    del img
    ns = convolve2d(np.ones(imgshape), kernel, mode="same")

    return np.sqrt((s2 - s**2 / ns) / ns)
