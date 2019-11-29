import unittest
import PDESolver
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim


class MyTestCase(unittest.TestCase):
    def test_tests(self):
        self.assertEqual(True, True)

    def test_image(self):
        image = cv2.imread("images/Apple.jpg")
        # Test it has three colours
        self.assertEqual(image.shape[2], 3)
        # Strip it to one colour:
        xs = image[:, :, 0]
        variance_start = np.var(image)
        # Add noise:
        err = 50
        xs = xs + np.random.randint(-err, err, xs.shape)
        # Check image is very different:
        # Using two measures of change. First is the similarity index (shows it is smoothing in the right way).
        similarity = ssim(image[:, :, 0], xs, data_range=xs.max() - xs.min())
        # Second is the variance. If this is decreasing it shows some smoothing is occurring.
        variance_1 = np.var(xs)
        self.assertTrue(similarity < 1.0)
        # Variance of initial image should be less than the variance of the one with the added noise
        self.assertTrue(variance_start < variance_1)
        # Improves?
        latest = PDESolver.spread_once(xs)
        similarity2 = ssim(image[:, :, 0], latest, data_range=latest.max() - latest.min())
        self.assertTrue(variance_1 > np.var(latest))
        self.assertTrue(similarity < similarity2)
        # Store the latest variance:
        variance_1 = np.var(latest)
        # Run for 20:
        for i in range(20):
            latest = PDESolver.spread_once(latest)
        # Check if similarity is increasing and variance is decreasing:
        err3 = ssim(image[:, :, 0], latest, data_range=latest.max() - latest.min())
        self.assertTrue(err3 > similarity2)
        self.assertTrue(variance_1 > np.var(latest))


if __name__ == '__main__':
    unittest.main()
