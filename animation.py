import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PDESolver import spread_colours

# Store Image as a numpy array:
file = "Apple"
im = cv2.imread("images/" + file + ".jpg")
xs = im.copy().astype(int)

# Add noise to the image:
added_error = 50
noise = np.random.randint(-added_error, added_error, xs.shape)
xs[1:-1, 1:-1] = xs[1:-1, 1:-1] + noise[1:-1, 1:-1]

# Initialize figure and plot initial image
fig = plt.figure()
image = plt.imshow(xs, animated=True)
ims = [[image]]

# Do image spreading
for i in range(20):
    # Only plot image after 5 runs
    for j in range(5):
        xs = spread_colours(xs)
    image = plt.imshow(xs, animated=True)
    ims.append([image])

# Create animation and save to gif
ani = animation.ArtistAnimation(fig, ims, interval=250, blit=True, repeat_delay=100)
ani.save("Results/" + file + "_animated_.gif", writer='imagemagick', fps=5)
