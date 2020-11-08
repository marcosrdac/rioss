import numpy as np


def overlapping_pool(img, whs=2, pool_func=np.std,
                     give_window=False, pool_func_kw={}):
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
    if rows % whs != 0:
        extra_row = 1
    if cols % whs != 0:
        extra_col = 1
    # pooling_layer = np.empty((pool_rows+extra_row, pool_cols+extra_col))
    pooling_layer = np.zeros((pool_rows+extra_row, pool_cols+extra_col))
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
    _m = overlapping_pool(m, 512, np.mean)

    plt.subplot(121)
    plt.imshow(m)
    plt.colorbar()
    plt.subplot(122)
    plt.imshow(_m)
    plt.colorbar()
    plt.show()
