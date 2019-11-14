# Copyright (C) 2019 Electronic Arts Inc.  All rights reserved.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
import glob
import random
import collections
import math

from helper import *
from layer import *
from consts import *

Examples = collections.namedtuple("Examples", "paths, inputs, targets, labels, count, steps_per_epoch")
Examples_inf = collections.namedtuple("Examples_inf",
                                      "paths, inputs, targets, labels, labels2, weight, count, steps_per_epoch")


def load_examples(input_dir, mode, lab_colorization,
                  which_direction, flip, scale_size, batch_size, png16bits, scop_name, mix_weight=False,
                  style_ref=False):
    """ Based on https://github.com/eric-guerin/pix2pix-tensorflow/blob/png16bits-support/pix2pix.py,
    see LICENSE file."""
    if input_dir is None or not os.path.exists(input_dir):
        raise Exception("input_dir does not exist")

    input_paths = glob.glob(os.path.join(input_dir, "*.jpg"))
    decode = tf.image.decode_jpeg
    if len(input_paths) == 0:
        input_paths = glob.glob(os.path.join(input_dir, "*.png"))
        decode = tf.image.decode_png

    if len(input_paths) == 0:
        raise Exception("input_dir contains no image files")

    def get_name(path):
        name, _ = os.path.splitext(os.path.basename(path))
        return name

    # If the image names are numbers, sort by the value rather than asciibetically
    # having sorted inputs means that the outputs are sorted in test mode.
    if all(get_name(path).isdigit() for path in input_paths):
        input_paths = sorted(input_paths, key=lambda path: int(get_name(path)))
    else:
        input_paths = sorted(input_paths)

    input_paths_t = tf.convert_to_tensor(input_paths, dtype=tf.string)
    if not style_ref:
        input_labels = [int(path.split('_')[-1][:-4]) for path in input_paths]
    else:
        input_labels = [0 for _ in input_paths]
    input_labels_t = tf.convert_to_tensor(input_labels, dtype=tf.int32)

    if mix_weight:
        input_labels2 = [int(path.split('_')[-2]) for path in input_paths]
        input_weight = [float(path.split('_')[-3]) for path in input_paths]
        input_labels2_t = tf.convert_to_tensor(input_labels2, dtype=tf.int32)
        input_weight_t = tf.convert_to_tensor(input_weight, dtype=tf.float32)
        input_queue = tf.train.slice_input_producer([input_paths_t, input_labels_t, input_labels2_t, input_weight_t],
                                                    shuffle=mode == "train")
    else:
        input_queue = tf.train.slice_input_producer([input_paths_t, input_labels_t], shuffle=mode == "train")

    with tf.name_scope(scop_name):
        if mix_weight:
            paths, contents, labels, labels2, weight = read_images_from_disk(input_queue, combine_weight=True)
        else:
            paths, contents, labels = read_images_from_disk(input_queue)

        if png16bits:
            raw_input = decode(contents, dtype=tf.uint16)
        else:
            raw_input = decode(contents)
        raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)

        assertion = tf.assert_equal(tf.shape(raw_input)[2], 3, message="image does not have 3 channels")
        with tf.control_dependencies([assertion]):
            raw_input = tf.identity(raw_input)

        raw_input.set_shape([None, None, 3])

        if lab_colorization:
            # load color and brightness from image, no B image exists here
            lab = rgb_to_lab(raw_input)
            L_chan, a_chan, b_chan = preprocess_lab(lab)
            a_images = tf.expand_dims(L_chan, axis=2)
            b_images = tf.stack([a_chan, b_chan], axis=2)
        else:
            # Break apart image pair and move to range [-1, 1]:
            width = tf.shape(raw_input)[1]  # [height, width, channels]
            a_images = preprocess(raw_input[:, :width // 2, :])
            b_images = preprocess(raw_input[:, width // 2:, :])

    if which_direction == "AtoB":
        inputs, targets = [a_images, b_images]
    elif which_direction == "BtoA":
        inputs, targets = [b_images, a_images]
    else:
        raise Exception("invalid direction")

    # Synchronize seed for image operations so that we do the same operations to both
    # input and output images.
    seed = random.randint(0, 2 ** 31 - 1)

    def transform(image):
        r = image
        if flip:
            r = tf.image.random_flip_left_right(r, seed=seed)

        # Area produces a nice downscaling, but does nearest neighbor for upscaling
        # assume we're going to be doing downscaling here.
        r = tf.image.resize_images(r, [scale_size, scale_size], method=tf.image.ResizeMethod.AREA)

        offset = tf.cast(tf.floor(tf.random_uniform([2], 0, scale_size - CROP_SIZE + 1, seed=seed)), dtype=tf.int32)
        if scale_size > CROP_SIZE:
            r = tf.image.crop_to_bounding_box(r, offset[0], offset[1], CROP_SIZE, CROP_SIZE)
        elif scale_size < CROP_SIZE:
            raise Exception("Scale size cannot be less than crop size.")
        return r

    with tf.name_scope("input_images"):
        input_images = transform(inputs)

    with tf.name_scope("target_images"):
        target_images = transform(targets)

    if mix_weight:
        paths_batch, inputs_batch, targets_batch, labels_batch, labels2_batch, weight_batch = tf.train.batch(
            [paths, input_images, target_images, labels, labels2, weight],
            batch_size=batch_size)
        steps_per_epoch = int(math.ceil(len(input_paths) / batch_size))

        return Examples_inf(
            paths=paths_batch,
            inputs=inputs_batch,
            targets=targets_batch,
            labels=labels_batch,
            labels2=labels2_batch,
            weight=weight_batch,
            count=len(input_paths),
            steps_per_epoch=steps_per_epoch,
        )
    else:
        paths_batch, inputs_batch, targets_batch, labels_batch = tf.train.batch(
            [paths, input_images, target_images, labels],
            batch_size=batch_size)
        steps_per_epoch = int(math.ceil(len(input_paths) / batch_size))

        return Examples(
            paths=paths_batch,
            inputs=inputs_batch,
            targets=targets_batch,
            labels=labels_batch,
            count=len(input_paths),
            steps_per_epoch=steps_per_epoch,
        )
