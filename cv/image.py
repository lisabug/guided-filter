import numpy as np


def to_32F(image):
    if image.max() > 1.0:
        image = image / 255.0
    return np.clip(np.float32(image), 0, 1)


def to_8U(image):
    if image.max() <= 1.0:
        image = image * 255.0
    return np.clip(np.uint8(image), 0, 255)


