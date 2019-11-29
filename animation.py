import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PDESolver import spread_colours

# Store Image as a numpy array:
file = "dog"
im = cv2.imread(file + ".jpg")
xs = im.copy().astype(int)

# Add noise to the image:
added_error = 50
noise = np.random.randint(-added_error, added_error, xs.shape)
xs[1:-1, 1:-1] = xs[1:-1, 1:-1] + noise[1:-1, 1:-1]

# Initialize figure
fig = plt.figure()
image = plt.imshow(xs, animated=True)
plt.show()
ims = [[image]]

for i in range(500):
    xs = spread_colours(xs)
    image = plt.imshow(xs, animated=True)
    ims.append([image])

ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat_delay=100)
#ani.save("animated_" + file + ".gif", writer='imagemagick', fps=200)
plt.show()