import unittest
import PDESolver
import numpy as np
import cv2


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
        error = np.mean((image[:, :, 0] - xs) ** 2)
        self.assertTrue(error > (err / 2) ** 2)
        # Improves?
        latest = PDESolver.spread_once(xs)
        error2 = np.mean((image[:, :, 0] - latest) ** 2)
        self.assertTrue(error > error2)
        # Run for 20:
        for i in range(20):
            latest = PDESolver.spread_once(latest)
        # Check if it is good:
        self.assertTrue(np.mean((image[:, :, 0] - latest) ** 2) < 10.0)


if __name__ == '__main__':
    unittest.main()
