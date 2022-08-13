import cv2
import numpy as np
import skimage.morphology as morp
from skimage.filters import rank


def grey_scale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def local_histo_equalize(image):
    kernel = morp.disk(30)
    img_local = rank.equalize(image, selem=kernel)
    return img_local


def image_normalize(image):
    image = np.divide(image, 255)
    return image


def preprocess(data):
    print("PREPROCESSING...")

    gray_image = list(map(grey_scale, data))
    equalize_images = list(map(local_histo_equalize, gray_image))
    n_data = data.shape
    processed_data = np.zeros((n_data[0], n_data[1], n_data[2]))
    for i, img in enumerate(equalize_images):
        processed_data[i] = image_normalize(img)
    processed_data = processed_data[..., None]

    return processed_data
