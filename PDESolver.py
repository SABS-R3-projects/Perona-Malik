import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation


k = 0.01

# Smooth,positive, non-increasing function g:
def func(x):
    return np.exp(-(x*k)**2)


# Calculates the divergence at each point of the matrix x
def divergence(x):
    grad = np.gradient(x)
    xs = np.gradient(grad[0])[0]
    ys = np.gradient(grad[1])[1]
    return xs + ys


# Helper function that normalizes before plotting and makes everything ints
def prepare_to_plot(xs):
    ranged = np.amax(xs) - np.amin(xs)
    return ((xs + np.amin(xs)) / ranged * 255).astype(int)


# Advance the matrix by one timestep (dt)
def spread_once(xs, g = func, dt=0.02):
    grad = np.gradient(xs)
    magnitude_of_grad = np.sqrt(grad[0] ** 2 + grad[1] ** 2)
    g_at_each_point = g(magnitude_of_grad)
    grad_of_g = np.gradient(g_at_each_point)
    xs[1:-1,1:-1] = xs[1:-1,1:-1] + dt*(divergence(xs)[1:-1,1:-1]*g_at_each_point[1:-1,1:-1]
                                        + grad[0][1:-1,1:-1] * grad_of_g[0][1:-1,1:-1]
                                        + grad[1][1:-1,1:-1] * grad_of_g[1][1:-1,1:-1])
    return xs


if __name__ == '__main__':

    # Store Image as a numpy array:
    im = cv2.imread("Smile.png")
    xs = im.copy()

    # Add noise to the image:
    noise = np.random.randint(-10, 10, xs.shape)
    xs = xs + noise

    plt.imshow(xs)
    plt.show()

    fig = plt.figure()
    #ims = []
    errs = []
    #ims.append([plt.imshow(xs, animated=True)])
    errs.append(np.mean((im-xs)**2))

    for i in range(100):
        xs[:, :, 0] = spread_once(xs[:, :, 0], func)
        xs[:, :, 1] = spread_once(xs[:, :, 1], func)
        xs[:, :, 2] = spread_once(xs[:, :, 2], func)
        #image = plt.imshow(xs, animated=True)
        #ims.append([image])
        errs.append(np.mean((im - xs) ** 2))

    #ani = animation.ArtistAnimation(fig, ims, interval=20, blit=True, repeat_delay=100)
    #plt.show()

    plt.plot(errs)
    plt.show()

    plt.imshow(xs)
    plt.show()

