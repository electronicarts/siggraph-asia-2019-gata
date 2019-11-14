# Copyright (C) 2019 Electronic Arts Inc.  All rights reserved.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from layer import *
from model_tuple import Model
from consts import *


def create_generator(generator_inputs, generator_outputs_channels, style_id, embedding_scope, a,
                     style_id2=None, weight=None, external_emb=None):
    layers = []

    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
    with tf.variable_scope("encoder_1"):
        output = gen_conv(generator_inputs, a.ngf, a.separable_conv)
        layers.append(output)

    layer_specs = [
        a.ngf * 2,  # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
        a.ngf * 4,  # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
        a.ngf * 8,  # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
        a.ngf * 8,  # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
        a.ngf * 8,  # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
        a.ngf * 8,  # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
        a.ngf * 8,  # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
    ]

    for out_channels in layer_specs:
        with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
            rectified = lrelu(layers[-1], 0.2)
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            convolved = gen_conv_sn(rectified, out_channels, a.separable_conv)
            output = batchnorm(convolved)
            layers.append(output)

    with tf.variable_scope(embedding_scope, reuse=True):
        style_embedding_table = tf.get_variable('style_embedding')

    if external_emb is None:
        style_embedded = tf.nn.embedding_lookup(style_embedding_table, style_id)
    else:
        style_embedded = external_emb
    style_embedded = tf.reshape(style_embedded, [-1, 1, 1, a.emb_dim])

    if not style_id2 is None:
        style_embedded_2 = tf.nn.embedding_lookup(style_embedding_table, style_id2)
        style_embedded_2 = tf.reshape(style_embedded_2, [-1, 1, 1, a.emb_dim])

        style_embedded = weight * style_embedded + (1 - weight) * style_embedded_2

    layer_specs = [
        (a.ngf * 8 * 2, 0.5),  # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
        (a.ngf * 8 * 2, 0.5),  # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
        (a.ngf * 8 * 2, 0.5),  # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
        (a.ngf * 8, 0.0),  # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
        (a.ngf * 4, 0.0),  # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
        (a.ngf * 2, 0.0),  # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
        (a.ngf, 0.0),  # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
    ]

    num_encoder_layers = len(layers)
    for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1
        with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
            if decoder_layer == 0:
                # first decoder layer doesn't have skip connections
                # since it is directly connected to the skip_layer
                input = layers[-1]
                rectified = tf.nn.relu(input)
                rectified = tf.concat([rectified, style_embedded], axis=3)

            else:
                input = tf.concat([layers[-1], layers[skip_layer]], axis=3)
                rectified = tf.nn.relu(input)

            # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
            output = gen_deconv_sn(rectified, out_channels, a.separable_conv)
            output = batchnorm(output)

            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)

            layers.append(output)

    # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
    with tf.variable_scope("decoder_1"):
        input = tf.concat([layers[-1], layers[0]], axis=3)
        rectified = tf.nn.relu(input)
        output = gen_deconv_sn(rectified, generator_outputs_channels, a.separable_conv)
        output = tf.tanh(output)
        layers.append(output)

    return layers[-1], style_embedded


def create_encoder(generator_inputs, discrim_targets, a):
    n_layers = 3
    layers = []

    # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
    input = tf.concat([generator_inputs, discrim_targets], axis=3)

    # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
    with tf.variable_scope("layer_1"):
        convolved = discrim_conv_sn(input, a.ndf, stride=2)
        rectified = lrelu(convolved, 0.2)
        layers.append(rectified)

    # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
    # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
    # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
    for i in range(n_layers):
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            out_channels = a.ndf * min(2 ** (i + 1), 8)
            stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
            convolved = discrim_conv_sn(layers[-1], out_channels, stride=stride)
            normalized = batchnorm(convolved)
            rectified = lrelu(normalized, 0.2)
            layers.append(rectified)

    # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
    with tf.variable_scope("layer_%d" % (len(layers) + 1)):
        convolved = discrim_conv_sn(rectified, out_channels=1, stride=1)
        normalized = batchnorm(convolved)
        output = lrelu(normalized, 0.2)
        layers.append(output)

    image_compressed = tf.reshape(layers[-1], [-1, 900])

    layer_specs = [
        a.ngf * 16,
        a.ngf * 16,
        a.ngf * 16,
    ]

    layers.append(image_compressed)

    for dense_layer, out_chal in enumerate(layer_specs):
        output = tf.layers.dense(inputs=layers[-1], units=out_chal, activation=None)
        output = lrelu(output, 0.2)
        layers.append(output)

    output = tf.layers.dense(inputs=layers[-1], units=a.emb_dim, activation=None)

    with tf.variable_scope("emb_result"):
        inf_emb = tf.reshape(output, [-1, a.emb_dim])

    return inf_emb


def create_model(inputs, targets, test_inputs, test_targets, labels, test_labels, a,
                 labels2=None, weight=None, external_emb=False):
    def create_discriminator(discrim_inputs, discrim_targets, style_id, embedding_scope, a):

        with tf.variable_scope(embedding_scope, reuse=True):
            style_embedding_table = tf.get_variable('style_embedding')

        style_embedded = tf.nn.embedding_lookup(style_embedding_table, style_id)
        style_embedded = tf.reshape(style_embedded, [-1, a.emb_dim])


        n_layers = 3
        layers = []
        activation = []

        # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
        input = tf.concat([discrim_inputs, discrim_targets], axis=3)

        # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
        with tf.variable_scope("layer_1"):
            convolved = discrim_conv_sn(input, a.ndf, stride=2)
            rectified = lrelu(convolved, 0.2)
            layers.append(rectified)
            activation.append(rectified)

        # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
        # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
        # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
        for i in range(n_layers):
            with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                out_channels = a.ndf * min(2 ** (i + 1), 8)
                stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
                convolved = discrim_conv_sn(layers[-1], out_channels, stride=stride)
                normalized = batchnorm(convolved)
                rectified = lrelu(normalized, 0.2)
                layers.append(rectified)
                activation.append(rectified)

        # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            convolved = discrim_conv_sn(rectified, out_channels=1, stride=1)
            normalized = batchnorm(convolved)
            output = lrelu(normalized, 0.2)
            layers.append(output)
            activation.append(output)

        image_compressed = tf.reshape(layers[-1], [-1, 900])

        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            output = tf.layers.dense(inputs=image_compressed, units=a.emb_dim, activation=None)
            output = lrelu(output, 0.2)
            layers.append(output)
            activation.append(output)

        with tf.variable_scope("dot_prod"):
            score = tf.reduce_sum(tf.multiply(output, style_embedded), 1, keep_dims=True)

        layer_specs = [
            a.ngf * 8,
            a.ngf * 4,
            a.ngf * 2,
        ]

        for dense_layer, out_chal in enumerate(layer_specs):
            with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                output = tf.layers.dense(inputs=output, units=out_chal, activation=None)
                output = lrelu(output, 0.2)
                layers.append(output)
                activation.append(output)

        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            output = tf.layers.dense(inputs=layers[-1], units=1, activation=None)
            layers.append(output)

        with tf.variable_scope("combine"):
            combine = score + output
            output = tf.sigmoid(combine)

        layers.append(tf.reshape(output, [-1, 1, 1, 1]))
        return layers[-1], activation

    with tf.variable_scope("embedding") as embedding_scope:
        style_emb_max = 786
        style_embedding_dim = a.emb_dim
        embeding_table = tf.get_variable(
            'style_embedding', [style_emb_max, style_embedding_dim], dtype=tf.float32, trainable=True)

    with tf.variable_scope("embedding_lables"):
        style_embed_labels = tf.stop_gradient(tf.nn.embedding_lookup(embeding_table, labels))
        overall_l2_norm = tf.stop_gradient(tf.reduce_mean(tf.norm(embeding_table, axis=1, ord=2)))

    if external_emb:
        external_emb = tf.placeholder(tf.float32, shape=(1, 1024), name='external_emb')
    else:
        external_emb = None

    with tf.variable_scope("generator"):
        out_channels = int(targets.get_shape()[-1])
        outputs, embed = create_generator(inputs, out_channels, labels, embedding_scope, a, labels2, weight, external_emb)

    with tf.variable_scope("generator", reuse=True):
        test_out_channels = int(test_targets.get_shape()[-1])
        test_outputs, test_embed = create_generator(test_inputs, test_out_channels, test_labels, embedding_scope, a, labels2, weight, external_emb)

    # create two copies of discriminator, one for real pairs and one for fake pairs
    # they share the same underlying variables
    with tf.name_scope("real_discriminator"):
        with tf.variable_scope("discriminator"):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_real, activation_real = create_discriminator(inputs, targets, labels, embedding_scope, a)

    with tf.name_scope("fake_discriminator"):
        with tf.variable_scope("discriminator", reuse=True):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_fake, activation_fake = create_discriminator(inputs, outputs, labels, embedding_scope, a)

    with tf.name_scope("real_encoder"):
        with tf.variable_scope("encoder_emb"):
            inf_embed_real = create_encoder(inputs, targets, a)

    with tf.name_scope("fake_encoder"):
        with tf.variable_scope("encoder_emb", reuse=True):
            inf_embed_fake = create_encoder(inputs, outputs, a)

    with tf.name_scope("discriminator_loss"):
        # minimizing -tf.log will try to get inputs to 1
        # predict_real => 1
        # predict_fake => 0
        discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))

    with tf.name_scope("feature_matching_loss"):
        loss_fm = tf.reduce_mean(tf.abs(activation_real[0] - activation_fake[0])) + \
                  tf.reduce_mean(tf.abs(activation_real[1] - activation_fake[1])) + \
                  tf.reduce_mean(tf.abs(activation_real[2] - activation_fake[2])) + \
                  tf.reduce_mean(tf.abs(activation_real[3] - activation_fake[3])) + \
                  tf.reduce_mean(tf.abs(activation_real[4] - activation_fake[4])) + \
                  tf.reduce_mean(tf.abs(activation_real[5] - activation_fake[5])) + \
                  tf.reduce_mean(tf.abs(activation_real[6] - activation_fake[6])) + \
                  tf.reduce_mean(tf.abs(activation_real[7] - activation_fake[7])) + \
                  tf.reduce_mean(tf.abs(activation_real[8] - activation_fake[8]))

    activation_num = 0.0
    for style_id in range(5):
        style_b, style_w, style_h, style_c = activation_real[style_id].shape
        activation_num += style_w * style_h * style_c
    with tf.name_scope("style_loss"):
        loss_style = 0.0  # No style loss in training.

    with tf.name_scope("generator_loss"):
        # predict_fake => 1
        # abs(targets - outputs) => 0
        gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))
        gen_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs))
        gen_loss_EMB = a.emb_weight * 0.5 * (tf.norm(inf_embed_real - style_embed_labels) + tf.norm(
            inf_embed_fake - style_embed_labels)) / overall_l2_norm
        gen_loss = gen_loss_GAN * a.gan_weight + gen_loss_L1 * a.l1_weight + gen_loss_EMB + a.fm_weight * loss_fm + a.style_weight * loss_style

    with tf.name_scope("discriminator_train"):
        discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
        discrim_tvars += [var for var in tf.trainable_variables() if var.name.startswith("embedding")]
        discrim_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
        discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
        discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)

    with tf.name_scope("encoder_train"):
        encoder_tvars = [var for var in tf.trainable_variables() if var.name.startswith("encoder_emb")]
        encoder_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
        encoder_grads_and_vars = encoder_optim.compute_gradients(gen_loss_EMB, var_list=encoder_tvars)
        encoder_train = encoder_optim.apply_gradients(encoder_grads_and_vars)

    with tf.name_scope("generator_train"):
        with tf.control_dependencies([discrim_train]):
            with tf.control_dependencies([encoder_train]):
                gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
                gen_tvars += [var for var in tf.trainable_variables() if var.name.startswith("embedding")]
                gen_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
                gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
                gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    update_losses = ema.apply([discrim_loss, gen_loss_GAN, gen_loss_L1, gen_loss_EMB, loss_fm, loss_style])

    global_step = tf.train.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step + 1)

    return Model(
        predict_real=predict_real,
        predict_fake=predict_fake,
        discrim_loss=ema.average(discrim_loss),
        discrim_grads_and_vars=discrim_grads_and_vars,
        gen_loss_GAN=ema.average(gen_loss_GAN),
        gen_loss_L1=ema.average(gen_loss_L1),
        gen_loss_EMB=ema.average(gen_loss_EMB),
        loss_fm=ema.average(loss_fm),
        loss_style=ema.average(loss_style),
        gen_grads_and_vars=gen_grads_and_vars,
        encoder_grads_and_vars=encoder_grads_and_vars,
        outputs=outputs,
        test_outputs=test_outputs,
        train=tf.group(update_losses, incr_global_step, gen_train),
        labels=labels,
        embedding=embed,
        external_emb=external_emb
    )
