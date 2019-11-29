import unittest
import PDESolver
import PDEfdsolver
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

image = cv2.imread("images/Apple.jpg")


class MyPDESolverTests(unittest.TestCase):

    def test_dummies(self):
        self.assertEqual(True, True)
        self.assertEqual(image.dtype, np.uint8)
        xs = PDESolver.spread_colours(image)
        self.assertEqual(xs.dtype, np.uint8)
        self.assertEqual(image.shape, xs.shape)
        # Test it has three colours
        self.assertEqual(image.shape[2], 3)

    # Checking first implementation
    def test_PDEfSolver(self):
        # Strip it to one colour:
        xs = image[:, :, 0]
        variance_start = np.var(xs)
        # Add noise:
        err = 50
        xs = xs + np.random.randint(-err, err, xs.shape)
        # Check image is very different:
        # Using two measures of change. First is the similarity index (shows it is smoothing in the right way).
        similarity = ssim(image[:, :, 0], xs, data_range=xs.max() - xs.min())
        # Second is the variance. If this is decreasing it shows some smoothing is occurring.
        variance_1 = np.var(xs)
        latest = PDEfdsolver.fd_smooth(xs)
        similarity2 = ssim(image[:, :, 0], latest, data_range=latest.max() - latest.min())
        self.assertTrue(variance_1 > np.var(latest))
        self.assertTrue(similarity < similarity2)
        # Store the latest variance:
        variance_1 = np.var(latest)
        # Run for 20:
        for i in range(3):
            latest = PDEfdsolver.fd_smooth(latest)
        # Check if similarity is increasing and variance is decreasing:
        err3 = ssim(image[:, :, 0], latest, data_range=latest.max() - latest.min())
        self.assertTrue(err3 > similarity2)
        self.assertTrue(variance_1 > np.var(latest))

    # Checking second implementation
    def test_PDESolver(self):
        # Strip it to one colour:
        xs = image[:, :, 0]
        variance_start = np.var(xs)
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
