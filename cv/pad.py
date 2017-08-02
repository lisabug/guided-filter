import numpy as np


def padding_constant(image, pad_size, constant_value=0):
    """
    Padding with constant value.

    Parameters
    ----------
    image: NDArray
        Image to padding. Only support 2D(gray) or 3D(color)
    pad_size: tuple
        Padding size for height and width axis respectively

    Returns
    -------
    ret: NDArray
        Image after padding
    """
    shape = image.shape
    assert len(shape) in [2, 3], 'image must be 2D or 3D'

    is_3D = True
    if len(shape) == 2:
        image = np.expand_dims(image, axis=2)
        shape = image.shape
        is_3D = False

    h, w = pad_size
    ret = np.zeros((shape[0]+2*h, shape[1]+2*w, shape[2]))
    ret[h:-h, w:-w, :] = image

    ret[:h, :, :] = constant_value
    ret[-h:, :, :] = constant_value
    ret[:, :w, :] = constant_value
    ret[:, -w:, :] = constant_value
    return ret if is_3D else np.squeeze(ret, axis=2)


def padding_reflect(image, pad_size):
    """
    Padding with reflection to image by boarder

    Parameters
    ----------
    image: NDArray
        Image to padding. Only support 2D(gray) or 3D(color)
    pad_size: tuple
        Padding size for height adn width axis respectively

    Returns
    -------
    ret: NDArray
        Image after padding
    """
    shape = image.shape
    assert len(shape) in [2, 3], 'image must be 2D or 3D'

    is_3D = True
    if len(shape) == 2:
        image = np.expand_dims(image, axis=2)
        shape = image.shape
        is_3D = False

    h, w = pad_size
    ret = np.zeros((shape[0]+2*h, shape[1]+2*w, shape[2]))

    for i in xrange(shape[0]+2*h):
        for j in xrange(shape[1]+2*w):
            if i < h:
                if j < w:
                    ret[i, j, :] = image[h-1-i, w-1-j, :]
                elif w <= j <= w + shape[1] - 1:
                    ret[i, j, :] = image[h-1-i, j-w, :]
                else:
                    ret[i, j, :] = image[h-1-i, w+2*shape[1]-1-j, :]
            elif h <= i <= h + shape[0] - 1:
                if j < w:
                    ret[i, j, :] = image[i-h, w-1-j, :]
                elif w <= j <= w + shape[1] - 1:
                    ret[i, j, :] = image[i-h, j-w, :]
                else:
                    ret[i, j, :] = image[i-h, w+2*shape[1]-1-j, :]
            else:
                if j < w:
                    ret[i, j, :] = image[h+2*shape[0]-1-i, w-1-j, :]
                elif w <= j <= w + shape[1] - 1:
                    ret[i, j, :] = image[h+2*shape[0]-1-i, j-w, :]
                else:
                    ret[i, j, :] = image[h+2*shape[0]-1-i, w+2*shape[1]-1-j, :]

    return ret if is_3D else np.squeeze(ret, axis=2)


def padding_reflect_101(image, pad_size):
    """
    Padding with reflection to image by boarder

    Parameters
    ----------
    image: NDArray
        Image to padding. Only support 2D(gray) or 3D(color)
    pad_size: tuple
        Padding size for height adn width axis respectively

    Returns
    -------
    ret: NDArray
        Image after padding
    """
    shape = image.shape
    assert len(shape) in [2, 3], 'image must be 2D or 3D'

    is_3D = True
    if len(shape) == 2:
        image = np.expand_dims(image, axis=2)
        shape = image.shape
        is_3D = False

    h, w = pad_size
    ret = np.zeros((shape[0]+2*h, shape[1]+2*w, shape[2]))

    for i in xrange(shape[0]+2*h):
        for j in xrange(shape[1]+2*w):
            if i < h:
                if j < w:
                    ret[i, j, :] = image[h-i, w-j, :]
                elif w <= j <= w + shape[1] - 1:
                    ret[i, j, :] = image[h-i, j-w, :]
                else:
                    ret[i, j, :] = image[h-i, w+2*shape[1]-2-j, :]
            elif h <= i <= h + shape[0] - 1:
                if j < w:
                    ret[i, j, :] = image[i-h, w-j, :]
                elif w <= j <= w + shape[1] - 1:
                    ret[i, j, :] = image[i-h, j-w, :]
                else:
                    ret[i, j, :] = image[i-h, w+2*shape[1]-2-j, :]
            else:
                if j < w:
                    ret[i, j, :] = image[h+2*shape[0]-2-i, w-j, :]
                elif w <= j <= w + shape[1] - 1:
                    ret[i, j, :] = image[h+2*shape[0]-2-i, j-w, :]
                else:
                    ret[i, j, :] = image[h+2*shape[0]-2-i, w+2*shape[1]-2-j, :]

    return ret if is_3D else np.squeeze(ret, axis=2)


def padding_edge(image, pad_size):
    """
    Padding with edge

    Parameters
    ----------
    image: NDArray
        Image to padding. Only support 2D(gray) or 3D(color)
    pad_size: tuple
        Padding size for height adn width axis respectively

    Returns
    -------
    ret: NDArray
        Image after padding
    """
    shape = image.shape
    assert len(shape) in [2, 3], 'image must be 2D or 3D'

    is_3D = True
    if len(shape) == 2:
        image = np.expand_dims(image, axis=2)
        shape = image.shape
        is_3D = False

    h, w = pad_size
    ret = np.zeros((shape[0]+2*h, shape[1]+2*w, shape[2]))

    for i in xrange(shape[0]+2*h):
        for j in xrange(shape[1]+2*w):
            if i < h:
                if j < w:
                    ret[i, j, :] = image[0, 0, :]
                elif w <= j <= w + shape[1] - 1:
                    ret[i, j, :] = image[0, j-w, :]
                else:
                    ret[i, j, :] = image[0, shape[1]-1, :]
            elif h <= i <= h + shape[0] - 1:
                if j < w:
                    ret[i, j, :] = image[i-h, 0, :]
                elif w <= j <= w + shape[1] - 1:
                    ret[i, j, :] = image[i-h, j-w, :]
                else:
                    ret[i, j, :] = image[i-h, shape[1]-1, :]
            else:
                if j < w:
                    ret[i, j, :] = image[shape[0]-1, 0, :]
                elif w <= j <= w + shape[1] - 1:
                    ret[i, j, :] = image[shape[0]-1, j-w, :]
                else:
                    ret[i, j, :] = image[shape[0]-1, shape[1]-1, :]

    return ret if is_3D else np.squeeze(ret, axis=2)

