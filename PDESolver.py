import numpy as np
import cv2
import matplotlib.pyplot as plt

# Smooth,positive, non-increasing function g:
def func(x):
    return np.exp(-x)


im = cv2.imread("dog.jpg")
print(type(im))
print(im.shape)
print(im[0,0])

plt.imshow(im[:,:,0], cmap = "gray")
plt.show()

