import cv2
from PDEfdsolver import *
from PDESolver import *


im = cv2.imread("images/cat.jpg")
im = im.copy()[125:225, 250:350, :]

dt = 0.1
ld = 0.01
iterations = 20
added_noise = 100

noise_im = add_noise(im, added_noise)

smoothed_im = fdsolver(noise_im, gfunc, dt, ld, iterations)
plot_results(im, noise_im, smoothed_im, added_noise)


