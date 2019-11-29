import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import optimize
import cma
from skimage.measure import compare_ssim as ssim

im = cv2.imread("Apple.jpg")
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
xs = im.copy()

def add_noise(xs):
    noise = np.random.randint(-50, 50, xs.shape)
    noisy_image = xs.astype(int) + noise
    return noisy_image

noisy_image = add_noise(xs)
start = noisy_image.copy()


# Smooth,positive, non-increasing function g:
def func(x, k=0.01):
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
def spread_once(xs, k=0.01, dt=0.02):
    grad = np.gradient(xs)
    magnitude_of_grad = np.sqrt(grad[0] ** 2 + grad[1] ** 2)
    g_at_each_point = np.exp(-(magnitude_of_grad)*k)
    grad_of_g = np.gradient(g_at_each_point)
    xs[1:-1,1:-1] = xs[1:-1,1:-1] + dt*(divergence(xs)[1:-1,1:-1]*g_at_each_point[1:-1,1:-1]
                                        + grad[0][1:-1,1:-1] * grad_of_g[0][1:-1,1:-1]
                                        + grad[1][1:-1,1:-1] * grad_of_g[1][1:-1,1:-1])
    return xs




def smoothing_function(k=0.01, dt =0.02):
    for i in range(10):
        print(i)
        noisy_image[:, :, 0] = spread_once(noisy_image[:, :, 0],k, dt)
        noisy_image[:, :, 1] = spread_once(noisy_image[:, :, 1],k, dt)
        noisy_image[:, :, 2] = spread_once(noisy_image[:, :, 2],k, dt)
    return noisy_image


def image_difference(k_array):
    filtered_image = smoothing_function(k_array[0])
    difference = 1 - ssim(noisy_image, filtered_image, multichannel=True,data_range=filtered_image.max() - filtered_image.min())
    return difference

def scoring_func():
    x0 = np.array([0.05])
    # A function that implements the CMA-ES algorithm on the noisy dataset
    minimization_sol, es = cma.fmin(image_difference, [0.03], 0.5)
    #minimization_sol = optimize.minimize(image_difference, x0, method= "Nelder-Mead", options = {'maxiter': 10})
    print(minimization_sol)


smooth_image = smoothing_function()
scoring_func()
plt.subplot(1, 3, 1)
plt.imshow(prepare_to_plot(im))
plt.subplot(1, 3, 2)
plt.imshow(prepare_to_plot(start))
plt.subplot(1, 3, 3)
plt.imshow(prepare_to_plot(smooth_image))
plt.show()