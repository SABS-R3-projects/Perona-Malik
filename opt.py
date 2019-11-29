import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage.metrics import structural_similarity as ssim
from functools import partial
from iminuit import Minuit
from PDESolver import spread_colours


def model(orig_image, noisy_image, k, dt, iterations=200):
    im = orig_image
    xs = noisy_image
    err = ssim(im, xs, data_range=xs.max() - xs.min(), multichannel=True)
    a = 0

    for i in range(iterations):
        xs = spread_colours(xs, dt=dt, ld=k)
        prev = err
        err = ssim(im, xs, data_range=xs.max() - xs.min(), multichannel=True)
        # If image gets worse stop iterating
        if prev > err:
            a = a + 1
        else:
            a = 0
        if (prev > err) and (a == 2):
            break

    return err


def create_images(image, added_error):
    # Store Image as a numpy array:
    im = cv2.imread(image)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    xs = im.copy()

    # Add noise to the image:
    noise = np.random.randint(-added_error, added_error, xs.shape)
    xs = xs + noise
    return im, xs


def minimise(call_number):
    im, noisy_im = create_images("images/Test-img.png", 30)

    def func_to_minimise(k, dt):
        return - model(im, noisy_im, k, dt, 50)

    # Using the Minuit optimiser:
    m = Minuit(func_to_minimise, k=0.01, dt=0.01, limit_dt=(0, 1))
    m.migrad(ncall=call_number)
    return m.values["k"], m.values["dt"]


if __name__ == "__main__":
    added_error = 20
    im, noisy_im = create_images("images/Test-img.png", added_error)

    k, dt = minimise(30)
    print('The optimal k is: ', k, ' and the optimal dt is: ', dt)
