import numpy as np


def overlapping_pool(img, whs=2, pool_func=np.std, extra=True,
                     give_window=False, pool_func_kw={}, last_dim=None, dtype=None):
    '''
    Function made to create pooling layers with any pooling function, which is
    run at windows with side `ws` and half-side `whs`. The windows overlap at
    every direction half image. If the number of image rows/columns are not
    multiple of `whs`, then the algorithm forces the creation of another
    row/column, so that all the image is viewed.

    :param img: 2D array to pool.
    :param whs: window half-side.
    :param pool_func: Pooling function to be used.
    :param pool_func_kw: dict of kwargs to be used for `pool_func`.
    '''
    ws = 2*whs
    rows, cols = img.shape
    assert not (ws > rows and ws > cols)
    # pooling layer allocation
    pool_rows = rows//ws + (rows-whs)//ws
    pool_cols = cols//ws + (cols-whs)//ws
    extra_col, extra_row = 0, 0
    if extra:
        if rows % whs != 0:
            extra_row = 1
        if cols % whs != 0:
            extra_col = 1
    outshape = [pool_rows+extra_row, pool_cols+extra_col]
    if last_dim is not None:
        outshape.append(last_dim)
    pooling_layer = np.empty(outshape, dtype=dtype)
    # pooling function calcylated at every window
    for xpi in range(pool_cols):
        xi = whs*xpi
        xf = xi + ws
        for ypi in range(pool_rows):
            yi = whs*ypi
            yf = yi + ws
            subimg = img[yi:yf, xi:xf]
            window_kw = {'window': ((yi, yf), (xi, xf))} if give_window else {}
            pooling_layer[ypi, xpi] = pool_func(
                subimg, **window_kw, **pool_func_kw)
        # if extra row, force the calculation on its cells
        if extra_row != 0:
            ypi += 1
            subimg = img[-ws:, xi:xf]
            window_kw = {'window': ((yi, yf), (xi, xf))} if give_window else {}
            pooling_layer[ypi, xpi] = pool_func(
                subimg, **window_kw, **pool_func_kw)
    # if extra col, force the calculation on its cells
    if extra_col != 0:
        xpi += 1
        xi = cols-ws
        xf = cols
        for ypi in range(pool_rows):
            yi = whs*ypi
            yf = yi + ws
            subimg = img[yi:yf, xi:xf]
            window_kw = {'window': ((yi, yf), (xi, xf))} if give_window else {}
            pooling_layer[ypi, xpi] = pool_func(subimg,
                                                **window_kw,
                                                **pool_func_kw)
        # extra row of extra column
        if extra_row != 0:
            ypi += 1
            yi = rows-ws
            yf = rows
            subimg = img[yi:yf, -ws:]
            window_kw = {'window': ((yi, yf), (xi, xf))} if give_window else {}
            pooling_layer[ypi, xpi] = pool_func(subimg,
                                            **window_kw,
                                            **pool_func_kw)
    return(pooling_layer)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np

    shape = (2204, 3555)
    m = np.arange(np.multiply(*shape)).reshape(shape)

    # one dimensional output for function np.mean
    _m = overlapping_pool(m, 512, np.mean)

    plt.figure(figsize=(8,4))
    plt.suptitle('One dimensional output')
    plt.subplot(121)
    plt.title('Original image')
    plt.imshow(m)
    plt.colorbar()
    plt.subplot(122)
    plt.title('Filtered image')
    plt.imshow(_m)
    plt.colorbar()
    plt.tight_layout()
    plt.show()


    # bidimensions output for function test_func
    def test_func(img):
        '''
        Bidimensinal output function.
        '''
        m = np.mean(img)
        return m, -m

    _m = overlapping_pool(m, 512, test_func, last_dim=2)

    plt.figure(figsize=(8,4))
    plt.suptitle('Bidimensional output')
    plt.subplot(121)
    plt.title('First layer of output (mean)')
    plt.imshow(_m[...,0])
    plt.colorbar()
    plt.subplot(122)
    plt.title('Second layer of output (-mean)')
    plt.imshow(_m[...,1])
    plt.colorbar()
    plt.tight_layout()
    plt.show()
