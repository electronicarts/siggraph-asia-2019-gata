# Copyright (C) 2019 Electronic Arts Inc.  All rights reserved.
import numpy as np
import cv2
import os
from psychopy.visual import filters
from os import walk

MAX_HEIGHT = 65536


def terrain_dilate(terrain, size):
    w, h = terrain.shape
    T = np.zeros((w + size * 2, h + size * 2), terrain.dtype)
    # T[size: size + w - 1, size: size + h - 1] = terrain
    # fill new cells using the nearest one
    new_w, new_h = T.shape
    # T[size:(w+size), size:(h+size)] = terrain
    for i in range(0, w + size * 2):
        for j in range(0, h + size * 2):
            I = i - size
            J = j - size
            if I < 0:
                I = 0
            elif I >= w:
                I = w - 1
            if J < 0:
                J = 0
            elif J >= h:
                J = h - 1
            T[i, j] = terrain[I, J]

    return T


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


def butter2d(image, cutoff, n):
    avr = np.average(image)
    std = np.std(image)
    image = (image - avr) / std
    img_freq = np.fft.fft2(image)
    lp_filt = filters.butter2d_lp(size=image.shape, cutoff=cutoff, n=n)
    img_filt = np.fft.fftshift(img_freq) * lp_filt
    s = np.real(np.fft.ifft2(np.fft.ifftshift(img_filt)))
    # s = np.clip(s, a_min=-1.0, a_max=1.0)
    return (s * std) + avr


def process_image(image_path, img_name, ratio, clip, proc):
    if image_path[-4:] != '.png':
        return
    terrain = read_height_field(image_path)
    l1, l2 = terrain.shape
    terrain = cv2.resize(terrain, (int(l1 * ratio), int(l2 * ratio)))
    l1, l2 = terrain.shape

    if clip != None:
        c1, c2, c3, c4 = clip
        terrain = terrain[int(l1 * c1):int(l1 * c2), int(l2 * c3):int(l2 * c4)]

    shift = (np.amin(terrain) + np.amax(terrain) - MAX_HEIGHT) / 2
    terrain -= shift

    d_s = 512
    terrain2 = terrain_dilate(terrain, d_s)
    terrain2 = butter2d(terrain2, 0.008, 3)
    terrain2 = terrain2[d_s:-d_s, d_s:-d_s]
    terrain2 = np.clip(terrain2, a_min=0, a_max=MAX_HEIGHT)

    l1, l2 = terrain.shape

    step_1 = 1024
    step_2 = 1024
    if (l1 / 1024) > 6:
        step_1 = int(l1 / 6)
        print('large image')
    if (l2 / 1024) > 6:
        step_2 = int(l2 / 6)
        print('large image')

    for i in range(0, l1, step_1):
        for j in range(0, l2, step_2):
            terrain_s = terrain[i:(i + 1024), j:(j + 1024)]
            w, h = terrain_s.shape
            if w < 1024 or h < 1024:
                continue
            terrain_a = np.zeros((w, h * 2), terrain_s.dtype)

            terrain_a[:, :h] = terrain_s
            terrain_a[:, h:] = terrain2[i:(i + 1024), j:(j + 1024)]
            figure = terrain_a.astype('uint16')
            print(proc + img_name + '_' + str(i) + '_' + str(j) + '.png')
            cv2.imwrite(proc + img_name + '_' + str(i) + '_' + str(j) + '.png', figure)


def main():
    overall = 0.5
    clip_dic = {'VistaHouse_8km_100cm_512m': (0.1, 1, 0, 1),
                'CloverButte_8km_100cm_2304m': (0, 0.9, 0.4, 1),
                'CraterLake_24km_100cm_2816m': (0.3, 0.9, 0.15, 0.75),
                'DepoeBay_4km_100cm_256m': (0, 1, 0.5, 1),
                'DetroitDam_8km_100cm_1600m': (0.3, 1, 0, 0.6),
                'HecetaHead_4km_100cm_512m': (0, 0.9, 0, 1),
                'PaintedHills_4km_50cm_1024m': (0, 1, 0.4, 1),
                }
    ratio_dic = {'ColumbiaRiver_16km_400cm_1536m': 4,
                 'EagleCap_16km_800cm_3072m': 4,
                 'PaintedHills_4km_50cm_1024m': 0.5,
                 }

    proc = './dataset/style_dataset/'
    if os.path.isdir(proc):
        raise Exception("Cannot overwrite the output folder")
    os.makedirs(proc)

    for (dirpath, dirnames, filenames) in walk('./dataset/raw_dataset'):
        for file in filenames:
            path = os.path.join(dirpath, file)
            print(path)
            ratio = 1
            if file[:-4] in ratio_dic.keys():
                print('ratio here:', file[:-4])
                ratio = ratio_dic[file[:-4]]
            ratio *= overall
            clip = None
            if file[:-4] in clip_dic.keys():
                print('clip here:', file[:-4])
                clip = clip_dic[file[:-4]]
            process_image(path, file[:-4], ratio, clip, proc=proc)


if __name__ == '__main__':
    main()
