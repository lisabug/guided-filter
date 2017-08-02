import cv2
import numpy as np
import itertools

from core.filter import GuidedFilter
from tools import visualize as vis
from cv.image import to_8U, to_32F



def test_gray():
    image = cv2.imread('data/cat.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    radius = [2, 4, 8]
    eps = [0.1**2, 0.2**2, 0.4**2]

    combs = list(itertools.product(radius, eps))

    vis.plot_single(image, title='origin')
    for r, e in combs:
        GF = GuidedFilter(image, radius=r, eps=e)
        vis.plot_single(GF.filter(image), title='r=%d, eps=%.2f' % (r, e))


def test_color():
    image = cv2.imread('data/Lenna.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    noise = (np.random.rand(image.shape[0], image.shape[1], 3) - 0.5) * 50
    image_noise = image + noise

    radius = [1, 2, 4]
    eps = [0.005]

    combs = list(itertools.product(radius, eps))

    vis.plot_single(to_32F(image), title='origin')
    vis.plot_single(to_32F(image_noise), title='noise')

    for r, e in combs:
        GF = GuidedFilter(image, radius=r, eps=e)
        vis.plot_single(to_32F(GF.filter(image_noise)), title='r=%d, eps=%.3f' % (r, e))


if __name__ == '__main__':
    test_gray()
    test_color()
