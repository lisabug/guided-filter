import cv2
import numpy as np

from core.filter import GuidedFilter
from tools import visualize as vis


image = cv2.imread('data/cat.png')


image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
# image = image.astype(np.float32)

# add noise
noise = (np.random.rand(image.shape[0], image.shape[1], 3) - 0.5) * 0
image_noise = image + noise

image_noise = cv2.cvtColor(image_noise.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)

GF = GuidedFilter(image, radius=8, eps=0.16)

output = GF.filter(image_noise)

# error = output - image_noise

vis.plot_multiple([image, image_noise, output],
                 ['image', 'input', 'output'],)
