import numpy as np

from cv.utils import blur2D


class GuidedFilter:
    """
    This is a factory class which builds guided filter
    according to the channel number of guided Input.
    The guided input could be gray image, color image,
    or multi-dimensional feature map.

    References:
        K.He, J.Sun, and X.Tang. Guided Image Filtering. TPAMI'12.
    """
    def __init__(self, I, radius, eps):
        """

        Parameters
        ----------
        I: NDArray with size of HWC or HW
        radius: float
        eps: float
        """
        if len(I.shape) == 2:
            self._Filter = GrayGuidedFilter(I, radius, eps)
        else:
            raise NotImplementedError

    def filter(self, p):
        if len(p.shape) == 2:
            return self._Filter.filter(p)
        elif len(p.shape) == 3:
            channels = p.shape[2]
            ret = np.zeros_like(p, dtype=np.float32)
            for c in channels:
                ret[:, :, c] = self._Filter.filter(p[:, :, c])
            return ret


class GrayGuidedFilter:
    """
    Specific guided filter for gray guided image.
    """
    def __init__(self, I, radius, eps):
        self.I = I
        self.radius = radius
        self.eps = eps

    def filter(self, p):
        """

        Parameters
        ----------
        p: 2D NDArray of filtering input

        Returns
        -------
        q: 2D NDArray of filtering output
        """
        # step 1
        meanI  = blur2D(I=self.I, r=self.radius)
        meanp  = blur2D(I=p, r=self.radius)
        corrI  = blur2D(I=self.I * self.I, r=self.radius)
        corrIp = blur2D(I=self.I * p, r=self.radius)
        # step 2
        varI   = corrI - meanI * meanI
        covIp  = corrIp - meanI * meanp
        # step 3
        a      = covIp / (varI + self.eps)
        b      = meanp - a * meanI
        # step 4
        meana  = blur2D(I=a, r=self.radius)
        meanb  = blur2D(I=b, r=self.radius)
        # step 5
        q = meana * self.I + meanb
        return q


# TODO
class MultiDimGuidedFilter:
    def __init__(self):
        pass

    def filter(self, p):
        pass