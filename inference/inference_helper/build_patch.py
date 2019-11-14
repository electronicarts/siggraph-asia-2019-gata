# Copyright (C) 2019 Electronic Arts Inc.  All rights reserved.
import numpy as np


def build_patch(patch_size):
    """
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
