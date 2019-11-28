import numpy as np
import cv2
import matplotlib.pyplot as plt
from medpy.filter.smoothing import anisotropic_diffusion
#from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim

img = np.random.uniform(size=(32,32))
img_filtered = anisotropic_diffusion(img)
im = cv2.imread("Apple.jpg")

xs = im.copy()

def prepare_to_plot(xs):
    ranged = np.amax(xs) - np.amin(xs)
    return ((xs + np.amin(xs)) / ranged * 255).astype(int)

# Add noise to the image:
def add_noise():
    xs = im.copy()
    noise = np.random.randint(-50, 50, xs.shape)
    xs = xs + noise
    return prepare_to_plot(xs)

def filter():
    img_filtered = anisotropic_diffusion(xs, niter = 45, option=3)
    return prepare_to_plot(img_filtered)

def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err

def plot_images():
    noisy_image = add_noise()
    filtered_image = filter()
    original_image = prepare_to_plot(im)
    #print(mse(original_image, filtered_image))
   # print(mse(noisy_image, filtered_image))
    #ssim_noise = ssim(original_image, filtered_image,data_range=filtered_image.max() - filtered_image.min())
   # print(ssim_noise)
    plt.subplot(1, 3, 1)
    plt.imshow(original_image)
    plt.xlabel("Original Image")
    plt.subplot(1, 3, 2)
    plt.imshow(noisy_image)
    plt.xlabel("Original Image + Noise")
    plt.subplot(1, 3, 3)
    plt.imshow(filtered_image)
    plt.xlabel("Filtered Image")
    plt.show()

plot_images()



