import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

img_file = 'img.jpg'
oil_pts_file = '/mnt/hdd/home/tmp/los/sar/blocks/oil_blocks.jpg'
lookalike_pts_file = '/mnt/hdd/home/tmp/los/sar/blocks/lookalike_blocks.jpg'
sea_pts_file = '/mnt/hdd/home/tmp/los/sar/blocks/sea_blocks.jpg'
terrain_pts_file = '/mnt/hdd/home/tmp/los/sar/blocks/terrain_blocks.jpg'
#files = [oil_img_file, lookalike_img_file, sea_pts_file, terrain_pts_file]


def first_channel(img):
    if len(img.shape) > 2:
        channel = img[:, :, 0]
    else:
        channel = img
    return(channel)


img = first_channel(np.array(Image.open(img_file)))
pts_img_bool = first_channel(np.array(Image.open(oil_pts_file))) > 127
ws = 512
skip = 1000
hws = int(ws//2)
#qws = int(ws//2)
qws = int(2*ws//3)

categories = ['oil', 'lookalike', 'sea', 'terrain']
colors = ['red', 'yellow', 'blue', 'green']


def sample_min_dist(bool_arr, dist, sample=.1, skip='auto'):
    if skip == 'auto':
        skip = dist
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


sampled_pts = sample_min_dist(pts_img_bool, qws)

fig, ax = plt.subplots()
ax.imshow(img, cmap='Greys_r')

# plotting all points (TOO MANY PLOTS)
# for c, pts in enumerate(pts_list):
#    for y, x in zip(pts[:, 0], pts[:, 1]):
#        # rect = patches.Rectangle((x-hws, y-hws), ws, ws,
#        #                         linewidth=1, edgecolor=colors[i], facecolor='none')
#        # ax.add_patch(rect)
#        ax.scatter(x, y, color=colors[c], facecolor='none')

for y, x in zip(sampled_pts[:, 0], sampled_pts[:, 1]):
    rect = patches.Rectangle((x-hws, y-hws), ws, ws,
                             linewidth=1, edgecolor='black', facecolor='none')
    ax.add_patch(rect)
    ax.scatter(x, y, color='black')

# plt.savefig('')
plt.show()
