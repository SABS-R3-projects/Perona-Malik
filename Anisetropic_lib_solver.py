import numpy as np
import cv2
import matplotlib.pyplot as plt
from medpy.filter.smoothing import anisotropic_diffusion
from skimage.measure import compare_ssim as ssim


img = np.random.uniform(size=(32,32))
im = cv2.imread("Apple.jpg")
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)# Changing the order of the color channel to RGB to get
                                        #a red apple rather than a blue one
xs = im.copy()

def prepare_to_plot(xs):
    """
    :param xs: input array of the image
    :return: Optimized array within the range of 0 to 255
    """
    ranged = np.amax(xs) - np.amin(xs)
    return ((xs + np.amin(xs)) / ranged * 255).astype(int)

def add_noise(xs):
    """
    A method used to add noise to an image
    :param xs: input array of the original image to be added noise to
    :return: An array corresponding to a noisy image
    """
    noise = np.random.randint(-30, 30, xs.shape)
    xs = xs + noise
    return prepare_to_plot(xs)

def filter(xs):
    """
    Method to filter noise out of an image.
    :param xs: an array corresponding to a noisy image
    :return: An array corresponding to a cleaner image
    """
    img_filtered = anisotropic_diffusion(xs, gamma = 0.1, kappa = 40, niter = 20, option=1)
    return prepare_to_plot(img_filtered)

def plot_images(xs):
    """
    Method used to display a panel of images before and after noise removal
    :param xs: An array corresponding to a clean image to be imported
    :return: A panel of the original image, image with noise added, and an image with the noise filtered
    """
    noisy_image = add_noise(xs)
    filtered_image = filter(noisy_image)
    original_image = prepare_to_plot(xs)
    #ssim calculates the structural similarity index between the noisy and the filtered image. The less the similarity index, the cleaner the filtered image
    ssim_noise = ssim(noisy_image, filtered_image,
                      multichannel=True,data_range=filtered_image.max()
                                                   - filtered_image.min())
    print(ssim_noise)
    plt.subplot(1, 3, 1)
    plt.imshow( original_image)
    plt.xlabel("Original Image")
    plt.subplot(1, 3, 2)
    plt.imshow(noisy_image)
    plt.xlabel("Original Image + Noise")
    plt.subplot(1, 3, 3)
    plt.imshow(filtered_image)
    plt.xlabel("Filtered Image")
    plt.show()

plot_images(xs)
