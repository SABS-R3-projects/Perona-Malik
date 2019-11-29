import cv2
from PDEfdsolver import *

im = cv2.imread("cat.jpg")
small_im = im.copy()[125:225, 250:350, :]

added_noise = 100

xs = add_noise(small_im, added_noise)
smoothed_im = smooth_pic(xs, dt=0.1, ld=100, iterations=20)

plot_results(small_im, xs, smoothed_im, added_noise)


