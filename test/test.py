import cv2
import numpy as np
import unittest

from cv.utils import blur2D


class TestBlurMethods(unittest.TestCase):

    def test_blur2D(self):
        # My implementation is different from cv2.blur..
        # Output on image boarding is different.
        # TODO
        I = np.array(range(1, 50)).reshape(7, 7).astype(np.float32)
        r = 1
        ret1 = blur2D(I, r)
        ret2 = cv2.blur(I, (3,3))
        print I
        print ret1
        print ret2
        self.assertTrue(np.array_equal(ret1, ret2))


if __name__ == '__main__':
    unittest.main()
