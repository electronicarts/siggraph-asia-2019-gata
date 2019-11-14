# Copyright (C) 2019 Electronic Arts Inc.  All rights reserved.
import numpy as np
import cv2
import math
from numpy import linalg as LA
import os
from os import walk

import random

MAX_HEIGHT = 65536
OUTPUT_LIST = './dataset/patched_dataset/style_list.txt'
TEST_RATIO = 0.01


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


def build_patch(patch_size):
    """
    Based on https://github.com/eric-guerin/terrain-amplification/blob/master/build_mask.m, see LICENSE file.
    Create a polynomial matrix of (patch_size, patch_size)
    :param patch_size: the diameter of the patch, an even integer
    :return: a square matrix
    """
    patch = np.zeros((patch_size, patch_size), np.double)
    radius = (patch_size - 1) * 0.5
    offset = 1.0 - 1.0 / patch_size
    for i in range(0, patch_size):
        for j in range(0, patch_size):
            x = (i - radius) / radius
            y = (j - radius) / radius
            val = 1 - offset * (x * x + y * y)
            if val < 0:
                val = 0
            patch[i, j] = val * val
    return patch


def build_dictionary(terrain, patch_size, offset, n_rotations, terrain_l, inx, output_dir):
    """
    Based on https://github.com/eric-guerin/terrain-amplification/blob/master/terrain_super_resolution.m,
    see LICENSE file.
    Build a dictionary of atoms from the terrain
    """
    offset_l, offset_w = offset
    patch_mask = build_patch(patch_size)

    d1 = int(math.floor((terrain.shape[0] - patch_size) / offset_l))
    d2 = int(math.floor((terrain.shape[1] - patch_size) / offset_w))

    print(d1 * d2 * n_rotations)

    if patch_mask.shape[1] != patch_size:
        print('Build dictionary failed! - patch mask needs to be square.')
        return

    mask_mean = np.zeros((d1, d2))

    range_max = []

    boundary = 0
    for i in range(0, d1):
        for j in range(0, d2):
            cur_patch = terrain[i * offset_l:i * offset_l + patch_size, j * offset_w:j * offset_w + patch_size]
            mean_h = np.mean(cur_patch)
            mask_mean[i, j] = mean_h
            cur_patch = terrain[i * offset_l:i * offset_l + patch_size, j * offset_w:j * offset_w + patch_size] - \
                        mask_mean[i, j] * np.full((patch_size, patch_size), 1)
            range_max.append(max(np.amax(cur_patch), -np.amin(cur_patch)))

    range_max.sort()
    current_r = range_max[-int(len(range_max) / 6)]
    print(current_r)

    print("Building terrain.")
    row_idx = 0
    step = 360 / n_rotations
    for i in range(boundary, d1 - boundary):
        for j in range(boundary, d2 - boundary):
            cur_patch = terrain[i * offset_l:i * offset_l + patch_size, j * offset_w:j * offset_w + patch_size] - \
                        mask_mean[i, j] * np.full((patch_size, patch_size), 1)
            cur_patch_l = terrain_l[i * offset_l:i * offset_l + patch_size, j * offset_w:j * offset_w + patch_size] - \
                          mask_mean[i, j] * np.full((patch_size, patch_size), 1)
            norm_v = LA.norm(cur_patch)
            if norm_v < 10000:
                continue
            for k in range(n_rotations):
                h = patch_size
                h2 = int(h / 2)
                M = cv2.getRotationMatrix2D((h2, h2), step * k, 1)
                rotated = cv2.warpAffine(cur_patch, M, (h, h))
                hr_figure = rotated
                hr_figure = np.rot90(hr_figure, (i + j) % 4)

                rotated_l = cv2.warpAffine(cur_patch_l, M, (h, h))
                lr_figure = rotated_l
                lr_figure = np.rot90(lr_figure, (i + j) % 4)

                figure = np.zeros([patch_size, patch_size * 2, 3])

                figure[:, :patch_size, 0] = lr_figure
                figure[:, patch_size:, 0] = hr_figure
                figure[:, :patch_size, 1] = lr_figure
                figure[:, patch_size:, 1] = hr_figure
                figure[:, :patch_size, 2] = lr_figure
                figure[:, patch_size:, 2] = hr_figure

                figure = figure * (MAX_HEIGHT / 3) / current_r
                figure += MAX_HEIGHT / 2
                if np.amax(figure) > MAX_HEIGHT or np.amin(figure) < 0:
                    print('Warning: np.amax(figure) > MAX_HEIGHT or np.amin(figure) < 0 does not hold.')
                    continue
                figure = figure.astype('uint16')
                if random.uniform(0, 1) < TEST_RATIO:
                    cv2.imwrite(
                        output_dir + 'validate/' + str(i) + '_' + str(j) + '_' + str(k) + '_style_' + str(inx) + '.png',
                        figure)
                else:
                    cv2.imwrite(
                        output_dir + 'train/' + str(i) + '_' + str(j) + '_' + str(k) + '_style_' + str(inx) + '.png',
                        figure)
            row_idx += 1

    return row_idx


def main():
    output_dir = './dataset/patched_dataset/'
    if os.path.isdir(output_dir):
        raise Exception("Cannot overwrite the output folder.")
    os.makedirs(output_dir)
    os.makedirs(output_dir + 'train/')
    os.makedirs(output_dir + 'validate/')

    if not OUTPUT_LIST is None:
        style_list_file = open(OUTPUT_LIST, "w+")

    for (dirpath, dirnames, filenames) in walk('./dataset/style_dataset/'):
        inx = 0
        for file in filenames:
            if file[-4:] != '.png':
                continue
            if not OUTPUT_LIST is None:
                file_name = file[:-4] + '.png'
                style_list_file.write('\'' + file_name + '\': \'' + str(inx) + '\',\n')
            path = os.path.join(dirpath, file)
            terrain_a = read_height_field(path)
            l, w = terrain_a.shape
            terrain = terrain_a[:, :int(w / 2)]
            terrain_l = terrain_a[:, int(w / 2):]
            print(np.amax(terrain), np.amin(terrain))

            l, w = terrain.shape

            patch_size = 256
            offset_l = min(int(l / 10), 256)
            offset_w = min(int(w / 10), 256)
            n_rotations = 1
            row_idx = build_dictionary(terrain, patch_size, (offset_l, offset_w), n_rotations, terrain_l, inx,
                                       output_dir)
            print(row_idx)
            inx += 1

    if not OUTPUT_LIST is None:
        style_list_file.close()


if __name__ == '__main__':
    main()
