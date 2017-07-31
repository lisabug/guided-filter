import numpy as np


def box_filter(I, r):
    """

    Parameters
    ----------
    I: NDArray
        Input should be 3D with format of HWC
    r: int
        radius of filter. kernel size = 2 * r + 1

    Returns
    -------
    ret: NDArray
        Output has same shape with input
    """
    I = I.astype(np.float32)
    shape = I.shape
    assert len(shape) in [2, 3], \
        "I should be NDArray of 2D or 3D, not %dD" % len(shape)
    is_3D = True

    if len(shape) == 2:
        I = np.expand_dims(I, axis=2)
        shape = I.shape
        is_3D = False

    (rows, cols, channels) = shape

    ret = np.zeros(shape=shape, dtype=np.float32)

    I_cum = np.cumsum(I, axis=0)
    ret[0:r+1, :, :] = I_cum[r:2*r+1, :, :]
    ret[r+1:rows-r, :, :] = I_cum[2*r+1:rows, :, :] - I_cum[0:rows-2*r-1, :, :]
    ret[rows-r:rows, :, :] = np.tile(I_cum[rows-1, :, :], (r, 1, 1)) - \
                          I_cum[rows-2*r-1:rows-r-1, :, :]

    I_cum = np.cumsum(ret, axis=1)
    ret[:, 0:r+1, :] = I_cum[:, r:2*r+1, :]
    ret[:, r+1:cols-r, :] = I_cum[:, 2*r+1:cols, :] - I_cum[:, 0:cols-2*r-1, :]
    ret[:, cols-r:cols, :] = np.tile(I_cum[:, cols-1, :], (r, 1, 1)).transpose((1, 0, 2)) - \
                          I_cum[:, cols-2*r-1:cols-r-1, :]

    return ret if is_3D else np.squeeze(ret, axis=2)


# TODO: add border type
def blur(I, r):
    """
    This method performs like cv2.blur().

    Parameters
    ----------
    I: NDArray
        Filtering input
    r: int
        Radius of blur filter

    Returns
    -------
    q: NDArray
        Blurred output of I.
    """
    ones = np.ones_like(I, dtype=np.float32)
    N = box_filter(ones, r)
    ret = box_filter(I, r)
    return ret / N
