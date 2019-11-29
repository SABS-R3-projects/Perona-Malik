import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage.metrics import structural_similarity as ssim
from multiprocessing import Pool
from functools import partial
from iminuit import Minuit


def spread_one_colour(image, g, dt0, k, dim):
    image[:, :, dim] = spread_once(image[:, :, dim], dt=dt0, g=g, k=k)
    return image[:, :, dim]


# Smooth,positive, non-increasing function g:
def func(x, k):
    return np.exp(-(x * k) ** 2)


# Calculates the divergence at each point of the matrix x
def divergence(x):
    grad = np.gradient(x)
    xs = np.gradient(grad[0])[0]
    ys = np.gradient(grad[1])[1]
    return xs + ys


# Helper function that normalizes before plotting
def prepare_to_plot(xs, normalize=True):
    if normalize:
        im = xs.copy()
        for i in range(3):
            ranged = np.amax(xs[:, :, i]) - np.amin(xs[:, :, i])
            im[:, :, i] = (xs[:, :, i] - np.amin(xs[:, :, i])) / ranged
        return im
    else:
        return xs / 255.0


# Advance the matrix by one timestep (dt)
def spread_once(xs, g=func, dt=0.01, k=0.01):
    grad = np.gradient(xs)
    magnitude_of_grad = np.sqrt(grad[0] ** 2 + grad[1] ** 2)
    g_at_each_point = g(magnitude_of_grad, k)
    grad_of_g = np.gradient(g_at_each_point)
    xs[1:-1, 1:-1] = xs[1:-1, 1:-1] + dt * (divergence(xs)[1:-1, 1:-1] * g_at_each_point[1:-1, 1:-1]
                                            + grad[0][1:-1, 1:-1] * grad_of_g[0][1:-1, 1:-1]
                                            + grad[1][1:-1, 1:-1] * grad_of_g[1][1:-1, 1:-1])
    return xs


def spread_colours(original_image, g=func, dt=0.01, k=0.01):
    image = original_image.copy()
    f = partial(spread_one_colour, image, g, dt, k)
    with Pool(3) as p:
        res = p.map(f, [0, 1, 2])
        for i in [0,1,2]:
            image[:, :, i] = res[i]

    return image


def model(orig_image, noisy_image, k, dt, iterations=200):
    im = orig_image
    xs = noisy_image
    err = ssim(im, xs, data_range=xs.max() - xs.min(), multichannel=True)
    err2 = np.var(im) / np.var(xs)
    errs = [[err, err2]]
    a = 0

    for i in range(iterations):
        xs = spread_colours(xs, dt=dt, k=k)
        prev = err
        err = ssim(im, xs, data_range=xs.max() - xs.min(), multichannel=True)
        err2 = np.var(im) / np.var(xs)
        errs.append([err, err2])

        if prev > err:
            a = a + 1
        else:
            a = 0
        if (prev > err) and (a == 2):
            print("Only did " + str(i) + " runs")
            print("Similarity value = " + str(err))
            break

    return xs, errs


def create_images(image, added_error):
    # Store Image as a numpy array:
    im = cv2.imread(image)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    xs = im.copy()

    # Add noise to the image:
    noise = np.random.randint(-added_error, added_error, xs.shape)
    xs = xs + noise
    return im, xs


def func_to_minimise(k, dt):
    im, noisy_im = create_images("images/Test-img.png", 30)
    img, errs = model(im, noisy_im, k, dt, 50)
    return errs[-1][0]


def minimise():
    m = Minuit(func_to_minimise, k=0.1, dt=0.1, limit_dt=(0,1))
    m.migrad(ncall=30)
    return m.values["k"], m.values["dt"]


if __name__ == "__main__":
    added_error = 20
    im, noisy_im = create_images("images/Test-img.png", added_error)

    xs, errs = model(im, noisy_im, 0.01, 0.01, iterations=50)


    ''' ===PLOTTING=== 
    plt.figure(figsize=(20, 20))
    plt.subplot(1, 3, 1)
    plt.imshow(noisy_im)
    plt.title("Image after +-" + str(added_error) + " of added noise")

    plt.subplot(1, 3, 2)
    plt.imshow(xs)
    plt.title("Image after smoothing")

    ax3 = plt.subplot(1, 3, 3)
    ax3.plot(errs)
    ax3.set_aspect("auto")
    plt.legend(["Similarity index", "Variance ratio"])
    plt.title("Evolution of similarity")
    plt.show()
'''
