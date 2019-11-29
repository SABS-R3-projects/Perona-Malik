import cv2
from PDEfdsolver import fdsolver, plot_results, gfunc, add_noise
from PDESolver import numpysolver
from Anisetropic_lib_solver import plot_images

"""
Give the path to the imread function and adjust the variables dt, ld and iterations.
"""
im = cv2.imread("images/cat.jpg")
im = im.copy()[125:225, 250:350, :]

dt = 0.1
ld = 0.01
iterations = 20

"""
If you want to add noise to your original image to see how good the smoothing is, do it here.
Otherwise, run noisy_im = im.copy() 
"""
added_noise = 100
noisy_im = add_noise(im, added_noise)
#noisy_im = im.copy()

"""
Don't change anything below here.
"""

# fd solver
smoothed_im = fdsolver(noisy_im, gfunc, dt, ld, iterations)
plot_results(im, noisy_im, smoothed_im, added_noise)

# fd solver using numpy
numpysolver(im, noisy_im, gfunc, dt, ld, iterations, added_noise)

# anisetropic lib solver
plot_images(im, noisy_im)

