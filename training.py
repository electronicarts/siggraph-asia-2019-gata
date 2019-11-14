# Copyright (C) 2019 Electronic Arts Inc.  All rights reserved.
"""
Based on a branch of pix2pix: [png16bits-support: https://github.com/eric-guerin/pix2pix-tensorflow]
Tensorflow port of Image-to-Image Translation with Conditional Adversarial Nets (pix2pix):
https://phillipi.github.io/pix2pix/
See LICENSE file.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os
import json
import random
import math
import time
from helper import *
from model import *
from consts import *
from dataloader import load_examples

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", default="dataset/patched_dataset/train/", help="path to folder containing images")
parser.add_argument("--output_dir", required=True, help="where to put output files")
parser.add_argument("--test_dir", default="dataset/patched_dataset/validate/",
                    help="path to folder containing test images")
parser.add_argument("--seed", type=int)
parser.add_argument("--checkpoint", default=None,
                    help="directory with checkpoint to resume training from or use for testing")

parser.add_argument("--max_steps", type=int, help="number of training steps (0 to disable)")
parser.add_argument("--max_epochs", type=int, help="number of training epochs")
parser.add_argument("--summary_freq", type=int, default=100, help="update summaries every summary_freq steps")
parser.add_argument("--progress_freq", type=int, default=50, help="display progress every progress_freq steps")
parser.add_argument("--trace_freq", type=int, default=0, help="trace execution every trace_freq steps")
parser.add_argument("--display_freq", type=int, default=0,
                    help="write current training images every display_freq steps")
parser.add_argument("--save_freq", type=int, default=10000, help="save model every save_freq steps, 0 to disable")

parser.add_argument("--separable_conv", action="store_true", help="use separable convolutions in the generator")
parser.add_argument("--aspect_ratio", type=float, default=1.0, help="aspect ratio of output images (width/height)")
parser.add_argument("--lab_colorization", action="store_true",
                    help="split input image into brightness (A) and color (B)")
parser.add_argument("--batch_size", type=int, default=1, help="number of images in batch")
parser.add_argument("--which_direction", type=str, default="AtoB", choices=["AtoB", "BtoA"])
parser.add_argument("--ngf", type=int, default=64, help="number of generator filters in first conv layer")
parser.add_argument("--ndf", type=int, default=64, help="number of discriminator filters in first conv layer")
parser.add_argument("--scale_size", type=int, default=256, help="scale images to this size before cropping to 256x256")
parser.add_argument("--flip", dest="flip", action="store_true", help="flip images horizontally")
parser.add_argument("--no_flip", dest="flip", action="store_false", help="don't flip images horizontally")
parser.set_defaults(flip=True)
parser.add_argument("--png16bits", dest="png16bits", action="store_true",
                    help="use png 16 bits images encoder and decoders")
parser.set_defaults(png16bits=False)
parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate for adam")
parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
parser.add_argument("--l1_weight", type=float, default=100.0, help="weight on L1 term for generator gradient")
parser.add_argument("--fm_weight", type=float, default=1.0,
                    help="weight on feature matching term for generator gradient")
parser.add_argument("--style_weight", type=float, default=0.0,
                    help="weight on feature matching term for generator gradient")
parser.add_argument("--gan_weight", type=float, default=1.0, help="weight on GAN term for generator gradient")

# Export options
parser.add_argument("--output_filetype", default="png", choices=["png", "jpeg"])

parser.add_argument("--emb_dim", type=int, default=1024, help="embedding dim")

parser.add_argument("--emb_weight", type=float, default=1.0, help="weight on L2 loss for embedding encoder")

a = parser.parse_args()


def main():
    if a.seed is None:
        a.seed = random.randint(0, 2 ** 31 - 1)

    tf.set_random_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)

    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)

    for k, v in a._get_kwargs():
        print(k, "=", v)

    with open(os.path.join(a.output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(a), sort_keys=True, indent=4))

    examples = load_examples(a.input_dir, "train", a.lab_colorization,
                             a.which_direction, a.flip, a.scale_size, a.batch_size, a.png16bits, "load_images")

    test_examples = load_examples(a.test_dir, "train", a.lab_colorization,
                                  a.which_direction, a.flip, a.scale_size, a.batch_size, a.png16bits,
                                  "load_test_images")

    print("examples count = %d" % examples.count)
    print("test examples count = %d" % test_examples.count)

    # Inputs and targets are [batch_size, height, width, channels].
    model = create_model(examples.inputs, examples.targets, test_examples.inputs, test_examples.targets,
                         examples.labels, test_examples.labels, a)

    # Undo colorization splitting on images that we use for display/output.
    if a.lab_colorization:
        if a.which_direction == "AtoB":
            # Inputs is brightness, this will be handled fine as a grayscale image
            # need to augment targets and outputs with brightness.
            targets = augment(examples.targets, examples.inputs)
            outputs = augment(model.outputs, examples.inputs)
            test_targets = augment(test_examples.targets, test_examples.inputs)
            test_outputs = augment(model.test_outputs, test_examples.inputs)
            # Inputs can be deprocessed normally and handled as if they are single channel
            # grayscale images.
            inputs = deprocess(examples.inputs)
            test_inputs = deprocess(test_examples.inputs)
        elif a.which_direction == "BtoA":
            # Inputs will be color channels only, get brightness from targets.
            inputs = augment(examples.inputs, examples.targets)
            targets = deprocess(examples.targets)
            outputs = deprocess(model.outputs)
            test_inputs = augment(test_examples.inputs, test_examples.targets)
            test_targets = deprocess(test_examples.targets)
            test_outputs = deprocess(model.test_outputs)
        else:
            raise Exception("invalid direction")
    else:
        inputs = deprocess(examples.inputs)
        targets = deprocess(examples.targets)
        outputs = deprocess(model.outputs)
        test_inputs = deprocess(test_examples.inputs)
        test_targets = deprocess(test_examples.targets)
        test_outputs = deprocess(model.test_outputs)

    def convert(image):
        if a.aspect_ratio != 1.0:
            # Upscale to correct aspect ratio.
            size = [CROP_SIZE, int(round(CROP_SIZE * a.aspect_ratio))]
            image = tf.image.resize_images(image, size=size, method=tf.image.ResizeMethod.BICUBIC)

        if a.png16bits:
            return tf.image.convert_image_dtype(image, dtype=tf.uint16, saturate=True)
        else:
            return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)

    # Reverse any processing on images so they can be written to disk or displayed to user.
    with tf.name_scope("convert_inputs"):
        converted_inputs = convert(inputs)
        test_converted_inputs = convert(test_inputs)

    with tf.name_scope("convert_targets"):
        converted_targets = convert(targets)
        test_converted_targets = convert(test_targets)

    with tf.name_scope("convert_outputs"):
        converted_outputs = convert(outputs)
        test_converted_outputs = convert(test_outputs)

    with tf.name_scope("encode_images"):
        display_fetches = {
            "paths": examples.paths,
            "inputs": tf.map_fn(tf.image.encode_png, converted_inputs, dtype=tf.string, name="input_pngs"),
            "targets": tf.map_fn(tf.image.encode_png, converted_targets, dtype=tf.string, name="target_pngs"),
            "outputs": tf.map_fn(tf.image.encode_png, converted_outputs, dtype=tf.string, name="output_pngs"),
        }

    # Summaries:
    if not a.png16bits:
        with tf.name_scope("inputs_summary"):
            tf.summary.image("inputs", converted_inputs)

        with tf.name_scope("targets_summary"):
            tf.summary.image("targets", converted_targets)

        with tf.name_scope("outputs_summary"):
            tf.summary.image("outputs", converted_outputs)

        with tf.name_scope("test_inputs_summary"):
            tf.summary.image("test_inputs", test_converted_inputs)

        with tf.name_scope("test_targets_summary"):
            tf.summary.image("test_targets", test_converted_targets)

        with tf.name_scope("test_outputs_summary"):
            tf.summary.image("test_outputs", test_converted_outputs)

    else:
        with tf.name_scope("inputs_summary"):
            tf.summary.image("inputs", tf.image.convert_image_dtype(converted_inputs, dtype=tf.uint8))

        with tf.name_scope("targets_summary"):
            tf.summary.image("targets", tf.image.convert_image_dtype(converted_targets, dtype=tf.uint8))

        with tf.name_scope("outputs_summary"):
            tf.summary.image("outputs", tf.image.convert_image_dtype(converted_outputs, dtype=tf.uint8))

        with tf.name_scope("test_inputs_summary"):
            tf.summary.image("test_inputs", tf.image.convert_image_dtype(test_converted_inputs, dtype=tf.uint8))

        with tf.name_scope("test_targets_summary"):
            tf.summary.image("test_targets", tf.image.convert_image_dtype(test_converted_targets, dtype=tf.uint8))

        with tf.name_scope("test_outputs_summary"):
            tf.summary.image("test_outputs", tf.image.convert_image_dtype(test_converted_outputs, dtype=tf.uint8))

    with tf.name_scope("predict_real_summary"):
        tf.summary.image("predict_real", tf.image.convert_image_dtype(model.predict_real, dtype=tf.uint8))

    with tf.name_scope("predict_fake_summary"):
        tf.summary.image("predict_fake", tf.image.convert_image_dtype(model.predict_fake, dtype=tf.uint8))

    tf.summary.scalar("discriminator_loss", model.discrim_loss)
    tf.summary.scalar("generator_loss_GAN", model.gen_loss_GAN)
    tf.summary.scalar("generator_loss_L1", model.gen_loss_L1)
    tf.summary.scalar("generator_loss_EMB", model.gen_loss_EMB)
    tf.summary.scalar("generator_loss_fm", model.loss_fm)
    tf.summary.scalar("generator_loss_style", model.loss_style)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name + "/values", var)

    for grad, var in model.discrim_grads_and_vars + model.gen_grads_and_vars + model.encoder_grads_and_vars:
        tf.summary.histogram(var.op.name + "/gradients", grad)

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
        # Training:
        start = time.time()
        saver_dir = os.path.join(a.output_dir, "saver")
        if not os.path.exists(saver_dir):
            os.makedirs(saver_dir)
        graph_dir = os.path.join(a.output_dir, 'graph_proto')
        if not os.path.exists(graph_dir):
            os.makedirs(graph_dir)
        model_ckpt_dir = os.path.join(a.output_dir, 'ckpt')
        if not os.path.exists(model_ckpt_dir):
            os.makedirs(model_ckpt_dir)
        loss_file = open(os.path.join(a.output_dir, "loss.txt"), "w")
        for step in range(max_steps):
            def should(freq):
                return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)

            options = None
            run_metadata = None
            if should(a.trace_freq):
                options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

            fetches = {
                "train": model.train,
                "global_step": sv.global_step,
            }

            if should(a.progress_freq):
                fetches["discrim_loss"] = model.discrim_loss
                fetches["gen_loss_GAN"] = model.gen_loss_GAN
                fetches["gen_loss_L1"] = model.gen_loss_L1
                fetches["gen_loss_EMB"] = model.gen_loss_EMB
                fetches["loss_fm"] = model.loss_fm
                fetches["loss_style"] = model.loss_style
                fetches["labels"] = model.labels

            if should(a.summary_freq):
                fetches["summary"] = sv.summary_op

            if should(a.display_freq):
                fetches["display"] = display_fetches

            results = sess.run(fetches, options=options, run_metadata=run_metadata)

            if should(a.summary_freq):
                print("Recording summary.")
                sv.summary_writer.add_summary(results["summary"], results["global_step"])

            if should(a.display_freq):
                print("saving display images")
                filesets = save_images(results["display"], step=results["global_step"])
                append_index(filesets, step=True)

            if should(a.trace_freq):
                print("recording trace")
                sv.summary_writer.add_run_metadata(run_metadata, "step_%d" % results["global_step"])

            if should(a.progress_freq):
                # Global_step will have the correct step count if we resume from a checkpoint.
                train_epoch = math.ceil(results["global_step"] / examples.steps_per_epoch)
                train_step = (results["global_step"] - 1) % examples.steps_per_epoch + 1
                rate = (step + 1) * a.batch_size / (time.time() - start)
                remaining = (max_steps - step) * a.batch_size / rate
                print("progress  epoch %d  step %d  image/sec %0.1f  remaining %dm" % (
                    train_epoch, train_step, rate, remaining / 60))
                print("discrim_loss", results["discrim_loss"])
                print("gen_loss_GAN", results["gen_loss_GAN"])
                print("gen_loss_L1", results["gen_loss_L1"])
                print("gen_loss_EMB", results["gen_loss_EMB"])
                print("loss_fm", results["loss_fm"])
                print("loss_style", results["loss_style"])
                print("labels:", results["labels"])

                if should(a.save_freq):
                    print("saving model")
                    saver.save(sess, os.path.join(a.output_dir, "model"), global_step=sv.global_step)

            if sv.should_stop():
                break
        loss_file.close()


if __name__ == '__main__':
    main()
