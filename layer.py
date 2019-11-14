# Copyright (C) 2019 Electronic Arts Inc.  All rights reserved.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os
import json
import glob
import random
import collections
import math
import time
from shutil import copyfile
from itertools import chain


def discrim_conv(batch_input, out_channels, stride):
    padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
    return tf.layers.conv2d(padded_input, out_channels, kernel_size=4, strides=(stride, stride), padding="valid",
                            kernel_initializer=tf.random_normal_initializer(0, 0.02))


def gen_conv(batch_input, out_channels, separable_conv):
    """ [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels] """
    initializer = tf.random_normal_initializer(0, 0.02)
    if separable_conv:
        print('GEN-CONV-SEP.')
        return tf.layers.separable_conv2d(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same",
                                          depthwise_initializer=initializer, pointwise_initializer=initializer)
    else:
        print('GEN-CONV.')
        return tf.layers.conv2d(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same",
                                kernel_initializer=initializer)


def gen_deconv(batch_input, out_channels):
    """ [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels] """
    initializer = tf.random_normal_initializer(0, 0.02)
    if a.separable_conv:
        _b, h, w, _c = batch_input.shape
        resized_input = tf.image.resize_images(batch_input, [h * 2, w * 2],
                                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return tf.layers.separable_conv2d(resized_input, out_channels, kernel_size=4, strides=(1, 1), padding="same",
                                          depthwise_initializer=initializer, pointwise_initializer=initializer)
    else:
        print('GEN-DECONV.')
        _b, h, w, _c = batch_input.shape
        resized_input = tf.image.resize_images(batch_input, [h * 2, w * 2],
                                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return tf.layers.conv2d(resized_input, out_channels, kernel_size=4, strides=(1, 1), padding="same",
                                kernel_initializer=initializer)


def spectral_norm(w, iteration=1):
    """ Implemented as in https://github.com/taki0112/Spectral_Normalization-Tensorflow/blob/master/spectral_norm.py
    See LICENSE file. """
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        # Power iteration: Usually iteration = 1 will be enough
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm


def discrim_conv_sn(batch_input, out_channels, stride):
    b, h, w, c = batch_input.shape
    w = tf.get_variable("kernel", shape=[4, 4, c, out_channels], initializer=tf.random_normal_initializer(0, 0.02))
    b = tf.get_variable("bias", [out_channels], initializer=tf.constant_initializer(0.0))

    padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")

    return tf.nn.conv2d(input=padded_input, filter=spectral_norm(w), strides=[1, stride, stride, 1],
                        padding="VALID") + b


def gen_conv_sn(batch_input, out_channels, separable_conv):
    """ [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels] """
    initializer = tf.random_normal_initializer(0, 0.02)
    if separable_conv:
        print('GEN-CONV-SEP.')
        return tf.layers.separable_conv2d(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same",
                                          depthwise_initializer=initializer, pointwise_initializer=initializer)
    else:
        print('GEN-CONV.')
        b, h, w, c = batch_input.shape
        w = tf.get_variable("kernel", shape=[4, 4, c, out_channels], initializer=tf.random_normal_initializer(0, 0.02))
        b = tf.get_variable("bias", [out_channels], initializer=tf.constant_initializer(0.0))

        return tf.nn.conv2d(input=batch_input, filter=spectral_norm(w), strides=[1, 2, 2, 1], padding="SAME") + b


def gen_deconv_sn(batch_input, out_channels, separable_conv):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    if separable_conv:
        _b, h, w, _c = batch_input.shape
        resized_input = tf.image.resize_images(batch_input, [h * 2, w * 2],
                                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return tf.layers.separable_conv2d(resized_input, out_channels, kernel_size=4, strides=(1, 1), padding="same",
                                          depthwise_initializer=initializer, pointwise_initializer=initializer)
    else:
        print('GEN-DECONV.')
        _b, h, w, _c = batch_input.shape
        resized_input = tf.image.resize_images(batch_input, [h * 2, w * 2],
                                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        w = tf.get_variable("kernel", shape=[4, 4, _c, out_channels], initializer=tf.random_normal_initializer(0, 0.02))
        b = tf.get_variable("bias", [out_channels], initializer=tf.constant_initializer(0.0))

        return tf.nn.conv2d(input=resized_input, filter=spectral_norm(w), strides=[1, 1, 1, 1], padding="SAME") + b


def gram_matrix(feature_maps):
    """Computes the Gram matrix for a set of feature maps."""
    batch_size, height, width, channels = tf.unstack(tf.shape(feature_maps))
    denominator = tf.to_float(height * width)
    feature_maps = tf.reshape(
        feature_maps, tf.stack([batch_size, height * width, channels]))
    matrix = tf.matmul(feature_maps, feature_maps, adjoint_a=True)
    return matrix / denominator


def lrelu(x, a):
    with tf.name_scope("lrelu"):
        # Adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # This block appears to have 2 inputs on the graph unless we do this:
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


def batchnorm(inputs):
    return tf.layers.batch_normalization(inputs, axis=3, epsilon=1e-5, momentum=0.1, training=True,
                                         gamma_initializer=tf.random_normal_initializer(1.0, 0.02))
