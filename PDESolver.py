import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage.metrics import structural_similarity as ssim

k = 0.01
dt = 0.01

# Smooth,positive, non-increasing function g:
def func(x):
    return np.exp(-(x * k) ** 2)


# Calculates the divergence at each point of the matrix x
def divergence(x):
    grad = np.gradient(x)
    xs = np.gradient(grad[0])[0]
    ys = np.gradient(grad[1])[1]
    return xs + ys


# Helper function that normalizes before plotting
def prepare_to_plot(xs, normalize = True):
    if normalize:
        im = xs.copy()
        for i in range(3):
            ranged = np.amax(xs[:, :, i]) - np.amin(xs[:, :, i])
            im[:, :, i] = (xs[:, :, i] - np.amin(xs[:, :, i])) / ranged
        return im
    else:
        return xs/255.0


# Advance the matrix by one timestep (dt)
def spread_once(xs, g=func, dt=dt):
    grad = np.gradient(xs)
    magnitude_of_grad = np.sqrt(grad[0] ** 2 + grad[1] ** 2)
    g_at_each_point = g(magnitude_of_grad)
    grad_of_g = np.gradient(g_at_each_point)
    xs[1:-1, 1:-1] = xs[1:-1, 1:-1] + dt * (divergence(xs)[1:-1, 1:-1] * g_at_each_point[1:-1, 1:-1]
                                            + grad[0][1:-1, 1:-1] * grad_of_g[0][1:-1, 1:-1]
                                            + grad[1][1:-1, 1:-1] * grad_of_g[1][1:-1, 1:-1])
    return xs


if __name__ == '__main__':

    # Store Image as a numpy array:
    im = cv2.imread("dog.jpg")
    xs = im.copy()#.astype(float)

    # Add noise to the image:
    added_error = 50
    noise = np.random.randint(-added_error, added_error, xs.shape)
    xs = xs + noise

    # fig = plt.figure()
    # ims = []
    err = ssim(im, xs, data_range=xs.max() - xs.min(), multichannel=True)
    err2 = np.var(im) / np.var(xs)
    errs = [[err, err2]]
    a = 0

    for i in range(200):
        xs[:, :, 0] = spread_once(xs[:, :, 0], func)
        xs[:, :, 1] = spread_once(xs[:, :, 1], func)
        xs[:, :, 2] = spread_once(xs[:, :, 2], func)
        prev = err
        err = ssim(im, xs, data_range=xs.max() - xs.min(), multichannel=True)
        err2 = np.var(im) / np.var(xs)
        # image = plt.imshow(xs, animated=True)
        # ims.append([image])
        errs.append([err, err2])
        if prev > err:
            a = a + 1
        else:
            a = 0
        if (prev > err) and (a == 1):
            print("Only did " + str(i) + " runs")
            print("Similarity value = " + str(err))
            break

    # ani = animation.ArtistAnimation(fig, ims, interval=20, blit=True, repeat_delay=100)
    # plt.show()
    plt.figure(figsize=(20,20))
    plt.subplot(1, 3, 1)
    plt.imshow(im+noise)
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