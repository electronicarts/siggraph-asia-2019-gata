# Copyright (C) 2019 Electronic Arts Inc.  All rights reserved.
import numpy as np


def terrain_dilate(terrain, size):
    w, h = terrain.shape
    T = np.zeros((w + size * 2, h + size * 2), terrain.dtype)
    # Fill new cells using the nearest one
    new_w, new_h = T.shape
    for i in range(0, new_w):
        for j in range(0, new_h):
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
