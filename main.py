import cv2
from PDEfdsolver import *
from PDESolver import *
from pints_opt import *


im = cv2.imread("images/cat.jpg")
im = im.copy()[125:225, 250:350, :]

dt = 0.1
ld = 0.01
iterations = 20
added_noise = 100

noisy_im = add_noise(im, added_noise)

smoothed_im = fdsolver(noisy_im, gfunc, dt, ld, iterations)
plot_results(im, noisy_im, smoothed_im, added_noise)


xs, errs = model(noisy_im, 0.01, 0.01, iterations=50)
plt.figure(figsize=(20, 20))
plt.subplot(1, 3, 1)
plt.imshow(noisy_im)
plt.title("Image after +-" + str(added_noise) + " of added noise")

plt.subplot(1, 3, 2)
plt.imshow(xs)
plt.title("Image after smoothing")

ax3 = plt.subplot(1, 3, 3)
ax3.plot(errs)
ax3.set_aspect("auto")
plt.legend(["Similarity index", "Variance ratio"])
plt.title("Evolution of similarity")
plt.show()


