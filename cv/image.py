import numpy as np


def to_32F(image):
    if image.max() > 1.0:
        image = image / 255.0
    return np.clip(np.float32(image), 0, 1)


def to_8U(image):
    if image.max() <= 1.0:
        image = image * 255.0
    else:
        image = image.astype(np.float32)
    range_min = np.min(image)
    range_max = np.max(image)
    if range_max > range_min:
        image = (image - range_min) * (255.0 / (range_max - range_min))
    return np.clip(np.round(image).astype(np.uint8), 0, 255)


