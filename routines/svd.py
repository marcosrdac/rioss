import numpy as np
from scipy.sparse import dia_matrix


def svd_filter(img, explainance=.8, full_matrices=False):
    U, s, VT = np.linalg.svd(img, full_matrices=full_matrices)

    norm_s = s/np.sum(s)
    cum_norm_s = np.cumsum(norm_s)

    k = np.searchsorted(cum_norm_s, explainance, side='right')
    if k == 0: k = 1
    S = dia_matrix((s[:k], 0), shape=(s.size, s.size))
    return U @ S @ VT


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    n, m = 50, 50

    img = np.arange(m)[None,:] * np.ones((n,m))
    img += np.arange(n)[:,None] * np.ones((n,m))
    img = img**2
    noise = 5000 * np.random.rand(n,m)
    noisy_img = img + noise
    img_filt = svd_filter(noisy_img, explainance=.3)
    noise_filt = noisy_img - img_filt
    residue = img_filt - img

    plt.subplot(231)
    plt.title('original')
    plt.imshow(img)

    plt.subplot(232)
    plt.title('noise')
    plt.imshow(noise)

    plt.subplot(233)
    plt.title('noisy image')
    plt.imshow(noisy_img)

    plt.subplot(234)
    plt.title('filt. image')
    plt.imshow(img_filt)

    plt.subplot(235)
    plt.title('filt. noise')
    plt.imshow(noise_filt)

    plt.subplot(236)
    plt.title('filt. image - original')
    plt.imshow(residue)

    plt.show()
