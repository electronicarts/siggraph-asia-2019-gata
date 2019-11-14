# Copyright (C) 2019 Electronic Arts Inc.  All rights reserved.
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter


def read_height_field(image_file):
    """
    Convert a terrain image to gray scale height field.
    :param image_file: filename of a terrain image
    :return: a reshaped height field numpy array
    """
    image = cv2.imread(image_file, -1)
    if image is None:
        print("The following file does not exist:")
        print(image_file)
        return None
    if len(image.shape) == 3:
        # in case of RGB image
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return np.double(image)


def normalize_dict(D, D_refer):
    D_norm = np.zeros(D.shape, np.double)
    n = 0
    for j in range(0, D.shape[1]):
        norm_val = np.linalg.norm(D_refer[:, j], ord=2)
        if norm_val != 0:
            D_norm[:, n] = D[:, j] / norm_val
            n += 1
    return D_norm


def read_mask(mask_file):
    mask = read_height_field(mask_file)
    if mask is None:
        return mask
    mask = mask / np.amax(mask)
    return 1.0 * (mask < 0.5)


def apply_gaussian(I, kernel):
    return gaussian_filter(I, sigma=kernel)


def process_mask(mask_o, mode='gaussian', kernel=None):
    """
    Smooth the mask using (Gaussian) filter.
        Input:
            mask_o: A numpy array which represent an original mask to be processed.
                1.0 means fully apply the mask while 0.0 means no mask at all.
            mode: For now, 'gaussian' means gaussian filter the original mask,
                'original' means keep using the non-changed mask.
        Output:
            a numpy array with same shape of mask_o.
    """
    if kernel is None:
        kernel = 9  # Default, learend from experimentation (future hyperparameter).
    if mode == 'original':
        return mask_o
    elif mode == 'gaussian':
        mask_n = gaussian_filter(mask_o, sigma=kernel)
        mask_n = mask_n / np.amax(mask_n)
        return mask_n
