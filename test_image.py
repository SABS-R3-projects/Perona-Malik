import unittest
import PDESolver
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim


class MyTestCase(unittest.TestCase):
    def test_tests(self):
        self.assertEqual(True, True)

    def test_image(self):
        image = cv2.imread("Apple.jpg")
        # Test it has three colours
        self.assertEqual(image.shape[2], 3)
        # Strip it to one colour:
        xs = image[:, :, 0]
        # Add noise:
        err = 50
        xs = xs + np.random.randint(-err, err, xs.shape)
        # Check image is very different:
        error = ssim(image[:, :, 0], xs, data_range=xs.max() - xs.min())
        self.assertTrue(error < 1.0 )
        # Improves?
        latest = PDESolver.spread_once(xs)
        error2 = ssim(image[:, :, 0], latest, data_range=latest.max() - latest.min())
        self.assertTrue(error < error2)
        # Run for 20:
        for i in range(20):
            latest = PDESolver.spread_once(latest)
        # Check if it is good:
        err3 = ssim(image[:, :, 0], latest, data_range=latest.max() - latest.min())
        self.assertTrue(err3 > error2)


if __name__ == '__main__':
    unittest.main()
