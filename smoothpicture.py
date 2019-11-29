import cv2
from PDEfdsolver import fdsolver, plot_results, gfunc, add_noise
from PDESolver import numpysolver
from Anisetropic_lib_solver import plot_images


im = cv2.imread("images/cat.jpg")
im = im.copy()[125:225, 250:350, :]

dt = 0.1
ld = 0.01
iterations = 20
added_noise = 100

noisy_im = add_noise(im, added_noise)

smoothed_im = fdsolver(noisy_im, gfunc, dt, ld, iterations)
plot_results(im, noisy_im, smoothed_im, added_noise)

numpysolver(im, noisy_im, gfunc, dt, ld, iterations, added_noise)

plot_images(im, noisy_im)

