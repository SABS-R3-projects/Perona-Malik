import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage.metrics import structural_similarity as ssim
from multiprocessing import Pool
from functools import partial
import os

k = 0.01
dt = 0.02


def spread_one_colour(image, g, dt0, dim, ld=0.01):
    """
    Helper function for multiprocessing. Do not call!!
    Has to be defined at the top of the file for some reason
    """
    image[:, :, dim] = spread_once(image[:, :, dim], dt=dt0, g=g, ld=ld)
    return image[:, :, dim]


def func(x, k=0.01):
    """
    User defined, smooth, positive, non-increasing function g:
    :param x: Function input (float)
    :param k: User defined parameter. Should be less than one and larger than 0
    :return: Function output (float)
    """
    return np.exp(-(x * k) ** 2)


def divergence(x):
    """
    Calculates the divergence at each point of the matrix x.
    :param x: Input matrix.
    :return: Returns a matrix containing the divergence at each point of the input matrix
    """
    grad = np.gradient(x)
    xs = np.gradient(grad[0])[0]
    ys = np.gradient(grad[1])[1]
    return xs + ys

def spread_once(xs, g=func, dt=dt, ld=0.01):
    """
    Evolves the input matrix by one timestep (dt) using the Perona-Malik equation.
    :param xs: Input matrix to be evolved
    :param g: User defined function to be used in the Perona-Malik algorithm
    :param dt: Timestep (measure of amount of correction)
    :param ld: Parameter in the user defined function
    :return: Matrix after one timestep
    """
    grad = np.gradient(xs)
    magnitude_of_grad = np.sqrt(grad[0] ** 2 + grad[1] ** 2)
    g_at_each_point = g(magnitude_of_grad, ld)
    grad_of_g = np.gradient(g_at_each_point)
    xs[1:-1, 1:-1] = xs[1:-1, 1:-1] + dt * (divergence(xs)[1:-1, 1:-1] * g_at_each_point[1:-1, 1:-1]
                                            + grad[0][1:-1, 1:-1] * grad_of_g[0][1:-1, 1:-1]
                                            + grad[1][1:-1, 1:-1] * grad_of_g[1][1:-1, 1:-1])
    return xs


def spread_colours(original_image, g=func, dt=dt, ld=0.01):
    """
    Evolves each colour in an image by one timestep (dt) according to the Perona-Malik equation.
    :param original_image: Image to be evolved
    :param g: User defined function to be used in the Perona-Malik algorithm
    :param dt: Timestep (measure of amount of correction)
    :param ld: Parameter in the user defined function
    :return: Evolved image after one timestep
    """
    image = original_image.copy()
    f = partial(spread_one_colour, image, g, dt, ld=ld)
    with Pool(3) as p:
        res = p.map(f, [0, 1, 2])
        for i in [0, 1, 2]:
            image[:, :, i] = res[i]

    return image


def numpysolver(im, noisy_im, gfunc, dt, ld, iterations, error_added):
    """
    Smooth a 3d matrix with the Perona-Malik equation with gfunc, dt and ld over iterations number of times.

    :param im: np.array 3d matrix (image)
    :param noisy_im: np.array noised 3d matrix (image)
    :param gfunc: g function used
    :param dt: float time step used
    :param ld: float lambda used
    :param iterations: int number of iterations
    :param error_added: int error added to the noised image
    :return:
    """
    xs = noisy_im.copy().astype(int)

    start = xs

    err = ssim(im, xs, data_range=xs.max() - xs.min(), multichannel=True)
    err2 = np.var(im) / np.var(xs)
    errs = [[err, err2]]
    a = 0

    for i in range(iterations):
        xs = spread_colours(xs, gfunc, dt, ld)
        prev = err
        err = ssim(im, xs, data_range=xs.max() - xs.min(), multichannel=True)
        err2 = np.var(im) / np.var(xs)
        errs.append([err, err2])
        if prev > err:
            a = a + 1
        else:
            a = 0
        if (prev > err) and (a == 1):
            print("Only did " + str(i) + " runs")
            print("Similarity value = " + str(err))
            break

    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(start)
    plt.title("Image after +-" + str(error_added) + " of added noise")

    plt.subplot(1, 2, 2)
    plt.imshow(xs)
    plt.title("Image after smoothing with\nk=" + str(k) + " and timestep of dt = " + str(dt))
    plt.savefig(os.path.join('Results', 'numpy_solver.png'), format='png')

    plt.subplot(1, 1, 1)
    plt.plot(errs)
    plt.legend(["Similarity index", "Variance ratio"])
    plt.title("Evolution of similarity")
    plt.savefig(os.path.join('Results', 'numpy_solver_graph.png'),format='png')


if __name__ == '__main__':

    # Store Image as a numpy array:
    im = cv2.imread("images/Test-img.png")
    xs = im.copy().astype(int)

    # Add noise to the image:
    added_error = 50
    noise = np.random.randint(-added_error, added_error, xs.shape)
    xs[1:-1, 1:-1] = xs[1:-1, 1:-1] + noise[1:-1, 1:-1]
    start = xs

    err = ssim(im, xs, data_range=xs.max() - xs.min(), multichannel=True)
    err2 = np.var(im) / np.var(xs)
    errs = [[err, err2]]
    a = 0

    for i in range(200):
        xs = spread_colours(xs)
        prev = err
        err = ssim(im, xs, data_range=xs.max() - xs.min(), multichannel=True)
        err2 = np.var(im) / np.var(xs)
        errs.append([err, err2])
        if prev > err:
            a = a + 1
        else:
            a = 0
        if (prev > err) and (a == 1):
            print("Only did " + str(i) + " runs")
            print("Similarity value = " + str(err))
            break

    plt.figure(figsize=(20, 20))
    plt.subplot(1, 3, 1)
    plt.imshow(start)
    plt.title("Image after +-" + str(added_error) + " of added noise")

    plt.subplot(1, 3, 2)
    plt.imshow(xs)
    plt.title("Image after smoothing with\nk=" + str(k) + " and timestep of dt = " + str(dt))

    ax3 = plt.subplot(1, 3, 3)
    ax3.plot(errs)
    ax3.set_aspect("auto")
    plt.legend(["Similarity index", "Variance ratio"])
    plt.title("Evolution of similarity")
    plt.show()