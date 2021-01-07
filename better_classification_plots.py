from os import listdir, makedirs
from os.path import expanduser, isdir, splitext, join, basename
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl



CLASSIFICATION_CATEGORIES = [
    {
        'name': 'oil',
        'color': 'red',
    },
    {
        'name': 'sea',
        'color': 'blue',
    },
    {
        'name': 'terrain',
        'color': 'brown',
    },
    {
        'name': 'phyto',
        'color': 'forestgreen',
    },
    {
        'name': 'rain',
        'color': 'lightskyblue',
    },
    {
        'name': 'wind',
        'color': 'grey',
    },
    # {
    # 'name': 'lookalike',
    # 'color': 'yellow',
    # },
]


def discarray(filename, mode='r', dtype=float, shape=None, order='C'):
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
        arr = np.memmap(io, mode=mode, dtype=dtype, shape=shape, 
                        offset=offset, order=order)
    return(arr)

# cmap = plt.get_cmap('tab10', 5)
colors = [c['color'] for c in CLASSIFICATION_CATEGORIES]
cmap = mpl.colors.LinearSegmentedColormap.from_list('test', colors)

folder = '/mnt/hdd/home/tmp/los/data/maps/20201109_afternoon'

files = [join(folder, f) for f in listdir(folder) if f.endswith('.bin')]
# shuffle(files)

for f in files:
    img = discarray(f, dtype=int).copy()
    fig, ax = plt.subplots()
    # pcol = ax.pcolormesh(img, cmap=cmap)
    pcol = ax.imshow(img, cmap=cmap)

    boundaries = np.arange(0, len(CLASSIFICATION_CATEGORIES)+1)
    values = boundaries[:-1]
    ticks = values + 0.5
    cbar = plt.colorbar(pcol,
                        ax=ax,
                        boundaries=boundaries,
                        orientation='horizontal',
                        values=values,
                        ticks=boundaries)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(np.arange(len(CLASSIFICATION_CATEGORIES)))
    cbar.set_ticklabels([c['name'] for c in CLASSIFICATION_CATEGORIES])

    plt.show()
    # break
