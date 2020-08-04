import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


def normalize256(img):
    return(img/255)


img = normalize256(np.array(Image.open('img.jpg')))
pts_img = normalize256(np.array(Image.open('pts.png')))
ws = 512
hws = int(ws//2)
qws = hws

categories = ['oil', 'look-alike', 'sea', 'terrain']
colors = ['red', 'yellow', 'blue', 'green']
# intensity values to use in images (00{V/100} of HSV)
# it's useful to make a palete

limits = np.linspace(0, 1, len([0]+categories+[1]))[1:-1]  # [.2, .4, .6, .8]
pts_list = []
for i, limit in enumerate(limits):
    pts = np.where(np.isclose(pts_img, limit, rtol=.02))
    pts_list.append(np.array(pts).T.copy())
    del(pts)

fig, ax = plt.subplots()

ax.imshow(img, cmap='Greys_r')

for c, pts in enumerate(pts_list):
    for y, x in zip(pts[:, 0], pts[:, 1]):
        # rect = patches.Rectangle((x-hws, y-hws), ws, ws,
        #                         linewidth=1, edgecolor=colors[i], facecolor='none')
        # ax.add_patch(rect)
        ax.scatter(x, y, color=colors[c], facecolor='none')

accepted_list = [np.ones(arr.shape[0], dtype=bool) for arr in pts_list]

for c in range(len(categories)):
    pts = pts_list[c]
    accepted = accepted_list[c]
    N = pts.shape[0]
    if N != 0:
        kdt = KDTree(pts)
        for p in range(N):
            if accepted[p]:
                pt = pts[p, :]
                near_pts = kdt.query_ball_point(pt, qws)
                for q in near_pts:
                    if q != p:
                        accepted[q] = False

for c in range(len(categories)):
    pts = pts_list[c]
    accepted = accepted_list[c]
    for i, (y, x) in enumerate(zip(pts[:, 0], pts[:, 1])):
        if accepted[i]:
            rect = patches.Rectangle((x-hws, y-hws), ws, ws,
                                     linewidth=1, edgecolor='black', facecolor='none')
            ax.add_patch(rect)
            ax.scatter(x, y, color='black')

#plt.savefig('get_kdtree_only_file_heavy_png_pts.jpg')
plt.show()
