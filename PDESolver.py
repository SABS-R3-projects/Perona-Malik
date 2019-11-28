import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# Smooth,positive, non-increasing function g:
def func(x):
    return np.exp(-x**2)


# Calculates the divergence at each point of the matrix x
def divergence(x):
    grad = np.gradient(x)
    xs = np.gradient(grad[0])[0]
    ys = np.gradient(grad[1])[1]
    return np.sqrt(xs ** 2 + ys ** 2)


# Helper function that normalizes before plotting and makes everything ints
def prepare_to_plot(xs):
    ranged = np.amax(xs) - np.amin(xs)
    return ((xs + np.amin(xs)) / ranged * 255).astype(int)


# Advance the matrix by one timestep (dt)
def spread_once(xs, g = func, dt=0.1):
    grad = np.gradient(xs)
    magnitude_of_grad = grad[0] ** 2 + grad[1] ** 2
    g_at_each_point = g(magnitude_of_grad)
    grad_of_g = np.gradient(g_at_each_point)
    return xs + dt * (g_at_each_point * divergence(xs) + grad[0] * grad_of_g[0] + grad[1] * grad_of_g[1])


if __name__ == '__main__':

    # Store Image as a numpy array:
    im = cv2.imread("Apple.jpg")
    xs = im.copy()

    plt.imshow(prepare_to_plot(im))
    plt.show()

    # Add noise to the image:
    noise = np.random.randint(-50, 50, xs.shape)
    xs = xs + noise

    fig = plt.figure()
    ims = []
    ims.append([plt.imshow(xs, animated=True)])

    for i in range(250):
        xs[:, :, 0] = spread_once(xs[:, :, 0], func)
        xs[:, :, 1] = spread_once(xs[:, :, 1], func)
        xs[:, :, 2] = spread_once(xs[:, :, 2], func)
        im = plt.imshow(xs, animated=True)
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=20, blit=True, repeat_delay=100)
    plt.show()

