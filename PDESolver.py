import numpy as np
import cv2
import matplotlib.pyplot as plt

# Smooth,positive, non-increasing function g:
def func(x):
    return np.exp(-x)

def divergence(x):
    grad = np.gradient(x)
    xs = np.gradient(grad[0])[0]
    ys = np.gradient(grad[1])[1]
    return np.sqrt(xs**2 + ys**2)


# Writing diff equation
# THIS IS NOT EVEN CLOSE TO WORKING!!!!!!!!!
def spread_once(xs, g):
    grad = np.gradient(xs)
    mag_grad = grad[0]**2 + grad[1]**2
    g_at_each_point = g(mag_grad)
    grad_of_g = np.gradient(g_at_each_point)
    return g_at_each_point @ divergence(xs) + grad @ grad_of_g




    return 0

im = cv2.imread("dog.jpg")
print(type(im))
print(im.shape)
print(im[0,0])

plt.imshow(im[:,:,0], cmap = "gray")
plt.show()

