import matplotlib.pyplot as plt
import numpy as np


def plot_single(img, title=''):
    plt.figure()
    plt.title(title)
    plt.imshow(img, cmap='gray')
    plt.waitforbuttonpress()


def plot_multiple(imgs, titles=''):
    num_img = len(imgs)
    rows = (num_img + 1) / 2
    f, axarr = plt.subplots(rows, 2)
    for i, (img, title) in enumerate(zip(imgs, titles)):
        axarr[i/2, i%2].imshow(img.astype(np.uint8), cmap='gray')
        axarr[i/2, i%2].set_title(title)
    plt.show()