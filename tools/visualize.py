import numpy as np
import matplotlib.pyplot as plt


def plot_single(img, title=''):
    plt.title(title)
    plt.imshow(img, cmap='gray')
    plt.show()


def plot_multiple(imgs, titles=''):
    num_img = len(imgs)
    rows = (num_img + 1) / 2
    f, axarr = plt.subplots(rows, 2)
    for i, (img, title) in enumerate(zip(imgs, titles)):
        axarr[i/2, i%2].imshow(img, cmap='gray')
        axarr[i/2, i%2].set_title(title)
    plt.show()