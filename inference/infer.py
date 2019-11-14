# Copyright (C) 2019 Electronic Arts Inc.  All rights reserved.
""" Based on pix2pix: https://phillipi.github.io/pix2pix/, see LICENSE file. """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
import json
import random
import collections
import time
from helper import deprocess, augment, save_images, append_index
from dataloader import load_examples
from model import create_model, create_encoder
from consts import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

Examples_inf = collections.namedtuple("Examples_inf",
                                      "paths, inputs, targets, labels, labels2, weight, count, steps_per_epoch")


def inference_model(a, external_emb=None):
    if a.seed is None:
        a.seed = random.randint(0, 2 ** 31 - 1)

    tf.set_random_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)

    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)

    if a.checkpoint is None:
        raise Exception("checkpoint required for test mode")

    # Load some options from the checkpoint.
    options = {"which_direction", "ngf", "ndf", "lab_colorization"}
    with open(os.path.join(a.checkpoint, "options.json")) as f:
        for key, val in json.loads(f.read()).items():
            if key in options:
                print("loaded", key, "=", val)
                setattr(a, key, val)
    # Disable these features in test mode.
    a.scale_size = CROP_SIZE
    a.flip = False

    for k, v in a._get_kwargs():
        print(k, "=", v)

    with open(os.path.join(a.output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(a), sort_keys=True, indent=4))

    examples = load_examples(a.input_dir, "train", a.lab_colorization,
                             a.which_direction, a.flip, a.scale_size, a.batch_size, a.png16bits, "load_images",
                             mix_weight=True)
    test_examples = load_examples(a.test_dir, "train", a.lab_colorization,
                                  a.which_direction, a.flip, a.scale_size, a.batch_size, a.png16bits, "load_images",
                                  mix_weight=True)

    # Inputs and targets are [batch_size, height, width, channels].
    model = create_model(examples.inputs, examples.targets, test_examples.inputs, test_examples.targets,
                         examples.labels, test_examples.labels, a, examples.labels2, examples.weight,
                         external_emb=(not external_emb is None))

    # Undo colorization splitting on images that we use for display/output.
    if a.lab_colorization:
        if a.which_direction == "AtoB":
            # inputs is brightness, this will be handled fine as a grayscale image
            # need to augment targets and outputs with brightness
            targets = augment(examples.targets, examples.inputs)
            outputs = augment(model.outputs, examples.inputs)
            # inputs can be deprocessed normally and handled as if they are single channel
            # grayscale images
            inputs = deprocess(examples.inputs)
        elif a.which_direction == "BtoA":
            # inputs will be color channels only, get brightness from targets
            inputs = augment(examples.inputs, examples.targets)
            targets = deprocess(examples.targets)
            outputs = deprocess(model.outputs)
        else:
            raise Exception("invalid direction")
    else:
        inputs = deprocess(examples.inputs)
        targets = deprocess(examples.targets)
        outputs = deprocess(model.outputs)

    def convert(image):
        if a.aspect_ratio != 1.0:
            # Upscale to correct aspect ratio.
            size = [CROP_SIZE, int(round(CROP_SIZE * a.aspect_ratio))]
            image = tf.image.resize_images(image, size=size, method=tf.image.ResizeMethod.BICUBIC)

        if a.png16bits:
            return tf.image.convert_image_dtype(image, dtype=tf.uint16, saturate=True)
        else:
            return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)

    # reverse any processing on images so they can be written to disk or displayed to user
    with tf.name_scope("convert_inputs"):
        converted_inputs = convert(inputs)

    with tf.name_scope("convert_targets"):
        converted_targets = convert(targets)

    with tf.name_scope("convert_outputs"):
        converted_outputs = convert(outputs)

    with tf.name_scope("encode_images"):
        display_fetches = {
            "paths": examples.paths,
            "inputs": tf.map_fn(tf.image.encode_png, converted_inputs, dtype=tf.string, name="input_pngs"),
            "targets": tf.map_fn(tf.image.encode_png, converted_targets, dtype=tf.string, name="target_pngs"),
            "outputs": tf.map_fn(tf.image.encode_png, converted_outputs, dtype=tf.string, name="output_pngs"),
        }

    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    saver = tf.train.Saver(max_to_keep=5)

    logdir = a.output_dir if (a.trace_freq > 0 or a.summary_freq > 0) else None
    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)

    with sv.managed_session() as sess:
        print("parameter_count =", sess.run(parameter_count))

        if a.checkpoint is not None:
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            saver.restore(sess, checkpoint)

        max_steps = 2 ** 32
        if a.max_epochs is not None:
            max_steps = examples.steps_per_epoch * a.max_epochs
        if a.max_steps is not None:
            max_steps = a.max_steps

        # Testing: process the test data once
        start = time.time()
        max_steps = min(examples.steps_per_epoch, max_steps)
        display_fetches["labels"] = model.labels
        for step in range(max_steps):
            if external_emb is None:
                results = sess.run(display_fetches)
            else:
                results = sess.run(display_fetches,
                                   feed_dict={model.external_emb: external_emb})
            print("labels: ", results["labels"])
            filesets = save_images(results, output_dir=a.output_dir)
            for i, f in enumerate(filesets):
                print("evaluated image", f["name"])
            index_path = append_index(filesets, output_dir=a.output_dir)
        print("wrote index at", index_path)
        print("rate", (time.time() - start) / max_steps, 's')

    tf.reset_default_graph()


def inference_emb(a):
    if a.seed is None:
        a.seed = random.randint(0, 2 ** 31 - 1)

    tf.set_random_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)

    if a.checkpoint is None:
        raise Exception("checkpoint required for test mode")

    # load some options from the checkpoint.
    options = {"which_direction", "ngf", "ndf", "lab_colorization"}
    with open(os.path.join(a.checkpoint, "options.json")) as f:
        for key, val in json.loads(f.read()).items():
            if key in options:
                print("loaded", key, "=", val)
                setattr(a, key, val)
    # Disable these features in test mode:
    a.scale_size = CROP_SIZE
    a.flip = False

    for k, v in a._get_kwargs():
        print(k, "=", v)

    examples = load_examples(a.ref_dir, "train", a.lab_colorization,
                             a.which_direction, a.flip, a.scale_size, a.batch_size, a.png16bits, "load_images",
                             style_ref=True)

    # Inputs and targets are [batch_size, height, width, channels].
    with tf.variable_scope("encoder_emb"):
        ref_emb = create_encoder(examples.inputs, examples.targets, a)

    with tf.name_scope("encode_images"):
        emb_fetches = {
            "ref_emb": ref_emb,
        }

    saver = tf.train.Saver(max_to_keep=5)

    logdir = a.output_dir if (a.trace_freq > 0 or a.summary_freq > 0) else None
    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)

    with sv.managed_session() as sess:
        if a.checkpoint is not None:
            print("Loading model from checkpoint...")
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            saver.restore(sess, checkpoint)
            print("Done loading model from checkpoint.")

        max_steps = 2 ** 32
        if a.max_epochs is not None:
            max_steps = examples.steps_per_epoch * a.max_epochs
        if a.max_steps is not None:
            max_steps = a.max_steps

        # Testing: process the test data once.
        max_steps = min(examples.steps_per_epoch, max_steps)
        for _ in range(max_steps):
            results = sess.run(emb_fetches)

    tf.reset_default_graph()
    return results["ref_emb"]
