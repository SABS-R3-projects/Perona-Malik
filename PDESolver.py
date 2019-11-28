import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage.metrics import structural_similarity as ssim

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
    im = cv2.imread("Apple.jpg")
    xs = im.copy()

    # Add noise to the image:
    noise = np.random.randint(-50, 50, xs.shape)
    xs = xs + noise

    plt.imshow(xs)
    plt.show()

    fig = plt.figure()
    #ims = []
    err = ssim(im, xs, data_range=xs.max() - xs.min(), multichannel = True)
    errs = [err]

    for i in range(80):
        xs[:, :, 0] = spread_once(xs[:, :, 0], func)
        xs[:, :, 1] = spread_once(xs[:, :, 1], func)
        xs[:, :, 2] = spread_once(xs[:, :, 2], func)
        prev = err
        err = ssim(im, xs, data_range=xs.max() - xs.min(), multichannel = True)
        #image = plt.imshow(xs, animated=True)
        #ims.append([image])
        errs.append(err)
        #if(prev > err):
            #break

    #ani = animation.ArtistAnimation(fig, ims, interval=20, blit=True, repeat_delay=100)
    #plt.show()

    plt.plot(errs)
    plt.show()

    plt.imshow(xs)
    plt.show()

