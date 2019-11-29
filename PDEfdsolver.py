import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from skimage.metrics import structural_similarity as ssim

def gfunc(x, ld):
    """
    Standard g function used in the Perona-Malik equation.

    :param x: float
    :param ld: float variable changing the curve of the g function
    :return: float
    """
    return np.exp(-(x * ld) ** 2)


def c_fd1(matrix, x, y):
    """
    Central difference of a given x and y in matrix matrix.

    :param matrix: np.array matrix
    :param x: int x coordinate
    :param y: int y coordinate
    :return: The central difference of a given x and y in matrix matrix
    """
    x1 = (matrix[x + 1, y] - matrix[x - 1, y]) / 2 * 1
    y1 = (matrix[x, y + 1] - matrix[x, y - 1]) / 2 * 1

    return np.array([x1, y1])


def mag(vec):
    """
    Magnitude of a vector.

    :param vec: np.array vector
    :return: float magnitude of vector vec
    """
    return np.sqrt(np.sum(np.power(vec, 2)))


def c_fd2(matrix, x, y):
    """
    Second Central difference of a given x and y in matrix matrix.

    :param matrix: np.array matrix
    :param x: int x coordinate
    :param y: int y coordinate
    :return: The Second Central difference of a given x and y in matrix matrix
    """
    x1 = (matrix[x + 1, y] - 2 * matrix[x, y] + matrix[x - 1, y])
    y1 = (matrix[x, y + 1] - 2 * matrix[x, y] + matrix[x, y - 1])

    return x1 + y1


def fd_smooth(input_m2d, gfunc=gfunc, dt=0.1, ld=1):
    """
    Applies the Perona-Malik equation solved with finite differences on a 2d matrix.

    :param input_m2d: 2d matrix to apply the Perona-Malik equation on
    :param gfunc: g function to use in the Perona-Malik equation
    :param dt: float time step used
    :param ld: float lambda used
    :return: a 2d matrix smoothed once with the Perona-Malik equation
    """

    m2d = input_m2d.copy()
    msize = m2d.shape

    g_m2d = np.zeros(msize)
    n_m2d = m2d.copy().astype(float)

    for x in range(1, msize[0] - 1):
        for y in range(1, msize[1] - 1):
            absux = np.abs(mag(c_fd1(m2d, x, y)))
            g_m2d[x, y] = gfunc(absux, ld)

    for x in range(1, msize[0] - 1):
        for y in range(1, msize[1] - 1):
            DDu = c_fd2(m2d, x, y)
            Dg = c_fd1(g_m2d, x, y)
            Du = c_fd1(m2d, x, y)

            n_m2d[x, y] = ((g_m2d[x, y] * DDu + np.dot(Dg, Du)) * dt + m2d[x, y])

    return n_m2d


def fdsolver(input_m3d, gfunc=gfunc, dt=0.1, ld=0.01, iterations=20):
    """
    Smooth a 3d matrix with the Perona-Malik equation with gfunc, dt and ld over iterations number of times.

    :param input_m3d: np.array 3d matrix (image)
    :param gfunc: g function used
    :param dt: float time step used
    :param ld: float lambda used
    :param iterations: int number of iterations
    :return: 4d matrix of the 3d image smoothed over iterations number of times.
    """
    m3d = input_m3d.copy()

    m4d = np.zeros(m3d.shape + (iterations,))

    for niter in range(iterations):
        m3d[:, :, 0] = fd_smooth(m3d[:, :, 0], gfunc, dt, ld)
        m3d[:, :, 1] = fd_smooth(m3d[:, :, 1], gfunc, dt, ld)
        m3d[:, :, 2] = fd_smooth(m3d[:, :, 2], gfunc, dt, ld)

        m4d[:, :, :, niter] = m3d

    return m4d

def add_noise(real_im, noise=50):
    """
    Adds noise to an image. Does not add noise to the border values.

    :param real_im: np.array 3d image
    :param noise: int how much noise to add
    :return: np.array 3d noised image
    """

    noisy_im = real_im.copy().astype(int)
    noisy_im[1:-1,1:-1,:] = noisy_im[1:-1,1:-1,:] + np.random.randint(-noise, noise, noisy_im[1:-1,1:-1,:].shape)

    return noisy_im

def plot_results(real_im, noisy_im, smoothed_im, added_error):
    """
    Creates and saves figures visualizing the smoothing of the data and the best smoothed image.

    :param real_im: np.array 3d image
    :param noisy_im: np.array 3d noised image
    :param smoothed_im: np.array 4d smoothed images for every iteration
    :param added_error: int error added to the noised image
    :return: Visualizing figures and smoothed images
    """
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(real_im)
    plt.title("Original image.")

    plt.subplot(2, 2, 2)
    compare_im = noisy_im
    plt.imshow(compare_im)
    RMSD = p_error(real_im, compare_im)
    SSIM = ssim(real_im, compare_im, data_range=compare_im.max() - compare_im.min(), multichannel=True)
    plt.title("Image after +- {0} of added noise.\n RMSD: {1:.2f} and SSIM: {2:.2f}.".format(added_error, RMSD, SSIM))

    plt.subplot(2, 2, 3)
    compare_im = smoothed_im[:, :, :, 0]
    plt.imshow(compare_im.astype(int))
    RMSD = p_error(real_im, compare_im)
    SSIM = ssim(real_im, compare_im, data_range=compare_im.max() - compare_im.min(), multichannel=True)
    plt.title("After running 1 iteration.\n RMSD: {1:.2f} and SSIM: {2:.2f}.".format(added_error, RMSD, SSIM))

    smooth_scores = [
        ssim(real_im, smoothed_im[:, :, :, i], data_range=real_im.max() - real_im.min(), multichannel=True) for i
        in range(smoothed_im.shape[-1])]
    m = max(smooth_scores)
    smooth_idx = [i for i, j in enumerate(smooth_scores) if j == m][0]

    plt.subplot(2, 2, 4)
    compare_im = smoothed_im[:, :, :, smooth_idx]
    plt.imshow(compare_im.astype(int))
    RMSD = p_error(real_im, compare_im)
    SSIM = ssim(real_im, compare_im, data_range=compare_im.max() - compare_im.min(), multichannel=True)
    plt.title(
        "After running {3} iterations (best SSIM).\n RMSD: {1:.2f} and SSIM: {2:.2f}.".format(added_error, RMSD, SSIM,
                                                                                              smooth_idx))

    plt.savefig(os.path.join('Results', 'fd_solver.png'), format='png')

    plt.subplot(1, 1, 1)
    plt.plot(smooth_scores)
    plt.xlabel("Number of iterations")
    plt.ylabel("Similarity index")
    plt.title("Evolution of similarity index")

    plt.savefig(os.path.join('Results', 'fd_solver_graph.png'), format='png')

def p_error(pic1, pic2):
    """
    Calculates the rmsd between two images.

    :param pic1: np.array 3d image
    :param pic2: np.array 3d image
    :return: rmsd between pic1 and pic2
    """
    am = pic1 - pic2
    return np.sqrt(np.mean(np.power(am.reshape(-1), 2)))


if __name__ == '__main__':

    # Store Image as a numpy array:
    im = cv2.imread("images/cat.jpg")

    small_im = im.copy()[125:225, 250:350, :]

    added_noise = 100

    xs = add_noise(small_im, added_noise)

    smoothed_im = fdsolver(xs, gfunc, dt=0.1, ld=100, iterations=20)

    plot_results(small_im, xs, smoothed_im, added_noise)
