import numpy as np


def _blur2D(I, r):
    """
    Helper method of blur2D
    """
    (rows, cols) = I.shape
    ret = np.zeros_like(I, dtype=np.float32)

    I_cum = np.cumsum(I, axis=0)
    ret[0:r+1, :] = I_cum[r:2*r+1, :]
    ret[r+1:rows-r, :] = I_cum[2*r+1:rows, :] - I_cum[0:rows-2*r-1, :]
    ret[rows-r:rows, :] = np.tile(I_cum[rows-1, :], (r, 1)) - \
                          I_cum[rows-2*r-1:rows-r-1, :]

    I_cum = np.cumsum(ret, axis=1)
    ret[:, 0:r+1] = I_cum[:, r:2*r+1]
    ret[:, r+1:cols-r] = I_cum[:, 2*r+1:cols] - I_cum[:, 0:cols-2*r-1]
    ret[:, cols-r:cols] = np.tile(I_cum[:, cols-1], (r, 1)).T - \
                          I_cum[:, cols-2*r-1:cols-r-1]

    return ret


def blur2D(I, r):
    """
    This method performs like cv2.blur().

    Parameters
    ----------
    I: NDArray of 2D
    r: float, radius of blur filter

    Returns
    -------
    Blurred output of I.
    """
    eye = np.ones_like(I, dtype=np.float32)
    N = _blur2D(eye, r)
    ret = _blur2D(I, r)
    return ret / N
