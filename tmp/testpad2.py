from functions import *
import scipy.signal as scpsig

#fn = "little_test_image.png"
fn = "/home/marcosrdac/res/wal/favorites/beach_pastel.jpg"


arr = np.mean(np.array(Image.open(fn)), 2)
# arr = np.where(arr>127, 1., 0.)


#kernel = np.array([[1,   4, 1],
#                   [4, -20, 4],
#                   [1,   4, 1],])
kernel = np.array([[0, 0, 1, 0, 0],
                   [0, 0, 1, 0, 0],
                   [1, 1, 1, 1, 1],
                   [0, 0, 1, 0, 0],
                   [0, 0, 1, 0, 0], ])
filt = get_convolver(kernel)
# filt = get_mwa(7)
# filt = get_mrwa(4,3)
#filt = mwsd

convolved_mine = filt(arr)
convolved_scp = scpsig.convolve(arr, kernel, mode='same')

fig, axes = plt.subplots(2, 2)

axes[0, 0].set_title("Array")
axes[0, 0].imshow(arr)
axes[0, 1].set_title("Kernel")
axes[0, 1].imshow(kernel)
axes[1, 0].set_title("Scipy: Kernel∗Array")
axes[1, 0].imshow(convolved_scp)
axes[1, 1].set_title("Parallel w/ sym. pad: Kernel∗Array")
axes[1, 1].imshow(convolved_mine)
fig.tight_layout()
plt.show()
