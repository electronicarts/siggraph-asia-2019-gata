# Copyright (C) 2019 Electronic Arts Inc.  All rights reserved.
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
import os
from psychopy.visual import filters
import scipy.misc

MAX_HEIGHT = 2 ** 16 - 1


def butter2d(image, cutoff, n):
    img_freq = np.fft.fft2(image)
    lp_filt = filters.butter2d_lp(size=image.shape, cutoff=cutoff, n=n)
    img_filt = np.fft.fftshift(img_freq) * lp_filt
    return np.real(np.fft.ifft2(np.fft.ifftshift(img_filt)))


def build_dictionary_big(terrain, patch_mask, offset, d1, d2, n_rotations, save_dic=False, style_dic=None,
                         weight_dic=None):
    """
        Build a dictionary of atoms from the terrain
    """
    patch_size = patch_mask.shape[0]
    if patch_mask.shape[1] != patch_size:
        print('Build dictionary failed! - patch mask needs to be square.')
        return
    # mask mean
    mask_mean = np.zeros((d1, d2))
    for i in range(0, d1):
        for j in range(0, d2):
            cur_patch = terrain[i * offset:i * offset + patch_size, j * offset:j * offset + patch_size]
            mask_mean[i, j] = np.mean(cur_patch)

    # build terrain
    X = []
    row_idx = 0
    step = 360 / n_rotations

    range_max = []

    for i in range(0, d1):
        for j in range(0, d2):
            cur_patch = terrain[i * offset:i * offset + patch_size, j * offset:j * offset + patch_size] - \
                        mask_mean[i, j] * np.full((patch_size, patch_size), 1)
            range_max.append(np.amax(cur_patch))

    range_max.sort()

    current_r = range_max[-1]
    print(current_r)
    for i in range(0, d1):
        for j in range(0, d2):
            style_id, style_id2 = style_dic[(i, j)]
            weight = weight_dic[(i, j)]
            cur_patch = terrain[i * offset:i * offset + patch_size, j * offset:j * offset + patch_size] - \
                        mask_mean[i, j] * np.full((patch_size, patch_size), 1)
            for k in range(n_rotations):
                h = patch_size
                h2 = int(h / 2)
                M = cv2.getRotationMatrix2D((h2, h2), step * k, 1)
                rotated = cv2.warpAffine(cur_patch, M, (h, h))
                coeff_mat = rotated

                if save_dic:
                    hr_figure = coeff_mat
                    lr_figure = coeff_mat
                    figure = np.zeros([patch_size, patch_size * 2, 3])
                    figure[:, :patch_size, 0] = lr_figure
                    figure[:, patch_size:, 0] = hr_figure
                    figure[:, :patch_size, 1] = lr_figure
                    figure[:, patch_size:, 1] = hr_figure
                    figure[:, :patch_size, 2] = lr_figure
                    figure[:, patch_size:, 2] = hr_figure
                    sc = max(np.amax(figure), -np.amin(figure))
                    X.append(sc)
                    figure = figure * (MAX_HEIGHT / 3) / current_r
                    figure += MAX_HEIGHT / 2
                    figure = np.clip(figure, a_min=0, a_max=MAX_HEIGHT)
                    figure = figure.astype('uint16')
                    cv2.imwrite(
                        save_dic + str(i) + '_' + str(j) + '_style_' + str(weight) + '_' + str(style_id2) + '_' + str(
                            style_id) + '.png', figure)
                    # Can use cv2.imwrite to save figure as an image file

            row_idx += 1

    return X, mask_mean, current_r
