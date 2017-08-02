import cv2
import numpy as np
import unittest
import cv.smooth


class TestBoxFilter(unittest.TestCase):

    def test_box_filter_reflect_101(self):
        I = np.array(range(1, 50)).reshape(7, 7).astype(np.float32)
        r = 2
        ret1 = cv.smooth.box_filter(I, r, normalize=True)
        ret2 = cv2.blur(I, (5,5), borderType=cv2.BORDER_DEFAULT)
        self.assertTrue(np.array_equal(ret1, ret2))

    def test_box_filter_reflect(self):
        I = np.array(range(1, 50)).reshape(7, 7).astype(np.float32)
        r = 2
        ret1 = cv.smooth.box_filter(I, r, normalize=True, border_type='reflect')
        ret2 = cv2.blur(I, (5,5), borderType=cv2.BORDER_REFLECT)
        self.assertTrue(np.array_equal(ret1, ret2))

    def test_box_filter_edge(self):
        I = np.array(range(1, 50)).reshape(7, 7).astype(np.float32)
        r = 2
        ret1 = cv.smooth.box_filter(I, r, normalize=True, border_type='edge')
        ret2 = cv2.blur(I, (5,5), borderType=cv2.BORDER_REPLICATE)
        self.assertTrue(np.array_equal(ret1, ret2))

    def test_box_filter_zero(self):
        I = np.array(range(1, 50)).reshape(7, 7).astype(np.float32)
        r = 2
        ret1 = cv.smooth.box_filter(I, r, normalize=True, border_type='zero')
        ret2 = cv2.blur(I, (5,5), borderType=cv2.BORDER_CONSTANT)
        self.assertTrue(np.array_equal(ret1, ret2))


if __name__ == '__main__':
    unittest.main()
