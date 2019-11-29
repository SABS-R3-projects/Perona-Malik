import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from skimage.metrics import structural_similarity as ssim

def gfunc(x, ld):
    return np.exp(-(x * ld) ** 2)


def c_fd1(matrix, x, y):
    x1 = (matrix[x + 1, y] - matrix[x - 1, y]) / 2 * 1
    y1 = (matrix[x, y + 1] - matrix[x, y - 1]) / 2 * 1

    return np.array([x1, y1])


def mag(vec):
    return np.sqrt(np.sum(np.power(vec, 2)))


def c_fd2(matrix, x, y):
    x1 = (matrix[x + 1, y] - 2 * matrix[x, y] + matrix[x - 1, y])
    y1 = (matrix[x, y + 1] - 2 * matrix[x, y] + matrix[x, y - 1])

    return x1 + y1


def fd_smooth(input_m2d, gfunc=gfunc, dt=0.1, ld=1):

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

    m3d = input_m3d.copy()

    m4d = np.zeros(m3d.shape + (iterations,))

    for niter in range(iterations):
        m3d[:, :, 0] = fd_smooth(m3d[:, :, 0], gfunc, dt, ld)
        m3d[:, :, 1] = fd_smooth(m3d[:, :, 1], gfunc, dt, ld)
        m3d[:, :, 2] = fd_smooth(m3d[:, :, 2], gfunc, dt, ld)

        m4d[:, :, :, niter] = m3d

    return m4d

def add_noise(real_im, noise=50):

    noisy_im = real_im.copy().astype(int)
    noisy_im[1:-1,1:-1,:] = noisy_im[1:-1,1:-1,:] + np.random.randint(-noise, noise, noisy_im[1:-1,1:-1,:].shape)

    return noisy_im

def plot_results(real_im, noisy_im, smoothed_im, added_error):
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
