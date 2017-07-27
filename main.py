import cv2
import numpy as np

from core.filter import GuidedFilter
from tools import visualize as vis


image = cv2.imread('data/Lenna.png')


image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

# add noise
noise = (np.random.rand(image.shape[0], image.shape[1]) - 0.5) * 20
image_noise = image + noise


GF = GuidedFilter(image, radius=1, eps=1e-2)

output = GF.filter(image_noise)

vis.plot_multiple([image, image_noise, output],
                  ['image', 'input', 'output'])
