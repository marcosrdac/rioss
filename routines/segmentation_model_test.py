from time import time
from tqdm import tqdm
from os import listdir
from os.path import join, basename
import matplotlib.pyplot as plt
from functions import normalize01
import numpy as np
import boxcounting
from segmentation import segmentate
import semivariogram


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


dir = "/mnt/hdd/home/tmp/los/data/classification_blocks"
filenames = [join(dir, f) for f in listdir(dir)]


filename = filenames[0]

for filename in tqdm(filenames):
    img = discarray(filename)
    t0 = time()
    segmented = segmentate(img)
    elapsed = time()-t0
    print(elapsed)

    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(img)
    axes[0].set_title(f'SVFD: {semivariogram.fractal_dimension(img)}')
    axes[1].imshow(segmented)
    axes[1].set_title(f'BCFD: {boxcounting.fractal_dimension(segmented)}')
    plt.savefig(f"/mnt/hdd/home/tmp/los/figures/{basename(filename[:-4])}.jpg")
    # plt.show()
    plt.close(fig)
