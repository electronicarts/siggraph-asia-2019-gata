# Copyright (C) 2019 Electronic Arts Inc.  All rights reserved.
from collections import namedtuple

# Shared between training and inference.
Model = namedtuple("Model",
    """
    outputs, 
    test_outputs, 
    predict_real, 
    predict_fake, 
    discrim_loss, 
    discrim_grads_and_vars, 
    gen_loss_GAN, 
    gen_loss_L1, 
    gen_loss_EMB, 
    loss_fm, 
    loss_style, 
    gen_grads_and_vars, 
    encoder_grads_and_vars, 
    train, labels, 
    embedding,
    external_emb
    """)
