from unittest import TestCase
import numpy as np
from mango.images.images_functions import (
    resize_img,
    resize_inv_crop,
    resize_img_max_size,
    fix_size_img,
    overlay_two_image_v2,
)


class ValidationTests(TestCase):
    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_resize_img(self):
        img = 200 * np.ones(shape=(500, 250, 3), dtype=np.uint8)
        rs = resize_img(img, size=(250, 125))
        self.assertEqual(img.shape[0] / img.shape[1], rs.shape[0] / rs.shape[1])

    def test_resize_inv_crop(self):
        img = np.ones(shape=(500, 250, 3), dtype=np.uint8)
        img[:20, :50, 0] = 100
        # rs = resize_img(img, size=(250, 125))
        box = [0, 0, 25, 10]
        rs = resize_inv_crop(img, box, size=(250, 125))

        # the sum of blue pixels before and after crop must be equal
        self.assertEqual(img[img[:, :, 0] == 100].sum(), rs.sum())

    def test_resize_img_max_size_h(self):
        img = 200 * np.ones(shape=(500, 250, 3), dtype=np.uint8)
        rs = resize_img_max_size(img, size=(400, 400))
        self.assertEqual(rs.shape[0], 400)

    def test_resize_img_max_size_w(self):
        img = 200 * np.ones(shape=(250, 500, 3), dtype=np.uint8)
        rs = resize_img_max_size(img, size=(400, 400))
        self.assertEqual(rs.shape[1], 400)

    def test_fix_size_img_bad_size(self):
        img = 200 * np.ones(shape=(250, 200, 3), dtype=np.uint8)
        rs = fix_size_img(img, size=(400, 400))
        self.assertEqual(rs, None)

    def test_fix_size_img(self):
        img = 200 * np.ones(shape=(250, 200, 3), dtype=np.uint8)
        rs = fix_size_img(img, size=(250, 250))
        self.assertEqual(rs.shape, (250, 250, 3))

    def test_overlay_two_image_v2(self):
        imgA = np.zeros(shape=(50, 50, 3), dtype=np.uint8)
        imgA[:, :, :2] = 255
        imgB = np.zeros(shape=(50, 50, 3), dtype=np.uint8)
        imgB[:, :, 1] = 255
        res = overlay_two_image_v2(imgA, imgB, 0.5)
        self.assertTrue((res[:, :, 0] == 127).all() & (res[:, :, 1] == 255).all())
