import numpy as np
import cv2
import matplotlib.pyplot as plt


# Smooth,positive, non-increasing function g:
def func(x):
    return np.exp(-x)


# Calculates the divergence at each point of the matrix x
def divergence(x):
    grad = np.gradient(x)
    xs = np.gradient(grad[0])[0]
    ys = np.gradient(grad[1])[1]
    return np.sqrt(xs ** 2 + ys ** 2)


# Helper function that normalizes before plotting and makes everything ints
def prepare_to_plot(xs):
    ranged = np.amax(xs) - np.amin(xs)
    return (xs / ranged * 250).astype(int)


# Advance the matrix by one timestep (dt)
def spread_once(xs, g, dt=0.1):
    grad = np.gradient(xs)
    magnitude_of_grad = grad[0] ** 2 + grad[1] ** 2
    g_at_each_point = g(magnitude_of_grad)
    grad_of_g = np.gradient(g_at_each_point)
    return xs + dt * (g_at_each_point * divergence(xs) + grad[0] * grad_of_g[0] + grad[1] * grad_of_g[1])


# Store Image as a numpy array:
im = cv2.imread("dog.jpg")
xs = im.copy()

plt.imshow(prepare_to_plot(im))
plt.show()

# Add noise to the image:
noise = np.random.randint(-50, 50, xs.shape)
xs = xs + noise

plt.imshow(prepare_to_plot(xs))
plt.show()

# Perform the spreading on all the colours of the image for 30 timesteps:
for i in range(20):
    xs[:, :, 0] = spread_once(im[:, :, 0], func)
    xs[:, :, 1] = spread_once(im[:, :, 1], func)
    xs[:, :, 2] = spread_once(im[:, :, 2], func)

plt.imshow(prepare_to_plot(xs))
plt.show()
