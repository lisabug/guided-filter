import numpy as np

from pad import padding_constant
from pad import padding_edge
from pad import padding_reflect
from pad import padding_reflect_101


def box_filter(I, r, normalize=True, border_type='reflect_101'):
    """

    Parameters
    ----------
    I: NDArray
        Input should be 3D with format of HWC
    r: int
        radius of filter. kernel size = 2 * r + 1
    normalize: bool
        Whether to normalize
    border_type: str
        Border type for padding, includes:
        edge        :   aaaaaa|abcdefg|gggggg
        zero        :   000000|abcdefg|000000
        reflect     :   fedcba|abcdefg|gfedcb
        reflect_101 :   gfedcb|abcdefg|fedcba

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

    tmp = np.zeros(shape=(rows, cols+2*r, channels), dtype=np.float32)
    ret = np.zeros(shape=shape, dtype=np.float32)

    # padding
    if border_type == 'reflect_101':
        I = padding_reflect_101(I, pad_size=(r, r))
    elif border_type == 'reflect':
        I = padding_reflect(I, pad_size=(r, r))
    elif border_type == 'edge':
        I = padding_edge(I, pad_size=(r, r))
    elif border_type == 'zero':
        I = padding_constant(I, pad_size=(r, r), constant_value=0)
    else:
        raise NotImplementedError

    I_cum = np.cumsum(I, axis=0) # (rows+2r, cols+2r)
    tmp[0, :, :] = I_cum[2*r, :, :]
    tmp[1:rows, :, :] = I_cum[2*r+1:2*r+rows, :, :] - I_cum[0:rows-1, :, :]

    I_cum = np.cumsum(tmp, axis=1)
    ret[:, 0, :] = I_cum[:, 2*r, :]
    ret[:, 1:cols, :] = I_cum[:, 2*r+1:2*r+cols, :] - I_cum[:, 0:cols-1, :]
    if normalize:
        ret /= float((2*r+1) ** 2)

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
