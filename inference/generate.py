# Copyright (C) 2019 Electronic Arts Inc.  All rights reserved.
import cv2
import math
import numpy as np
import os

from inference.inference_helper.commons import read_height_field
from inference.inference_helper.build_patch import build_patch
from inference.inference_helper.build_dictionary_big import build_dictionary_big
from inference.inference_helper.terrain_dilate import terrain_dilate
from inference.infer import inference_model, inference_emb

from psychopy.visual import filters
import argparse

MAX_HEIGHT = 2 ** 16 - 1

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int)
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
parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate for adam")
parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
parser.add_argument("--l1_weight", type=float, default=100.0, help="weight on L1 term for generator gradient")
parser.add_argument("--fm_weight", type=float, default=1.0,
                    help="weight on feature matching term for generator gradient")
parser.add_argument("--style_weight", type=float, default=250.0,
                    help="weight on feature matching term for generator gradient")
parser.add_argument("--gan_weight", type=float, default=1.0, help="weight on GAN term for generator gradient")

# export options
parser.add_argument("--output_filetype", default="png", choices=["png", "jpeg"])

parser.add_argument("--emb_dim", type=int, default=1024, help="embedding dim")

parser.add_argument("--emb_weight", type=float, default=0.0, help="weight on L2 loss for embedding encoder")

a = parser.parse_args()


def butter2d(image, cutoff, n):
    avr = np.average(image)
    std = np.std(image)
    image = (image - avr) / std
    img_freq = np.fft.fft2(image)
    lp_filt = filters.butter2d_lp(size=image.shape, cutoff=cutoff, n=n)
    img_filt = np.fft.fftshift(img_freq) * lp_filt
    s = np.real(np.fft.ifft2(np.fft.ifftshift(img_filt)))
    return (s * std) + avr


def terrain_amplification_big(height_field, factor, isUint16=True, style_list=(111, 289), bound_w=300,
                              fix_ratio=None, external_emb=None, smooth_input=True):
    """
    Given a low resolution height field (HF), and a high resolution exemplar HF containing landform details,
    learn a high resolution terrain preserving the topological feature of the low res HF while presenting geological
    features of the high res HF.

    :return: a high resolution HF that can be imported to Blender.
    """
    patch_size = 64

    if not style_list is None:
        style_id_nn, style_id_nn2 = style_list
    elif not external_emb is None:
        style_id_nn, style_id_nn2 = 0, 0

    h_lr, w_lr = height_field.shape[0], height_field.shape[1]

    height_field = height_field[:int(h_lr / 256) * 256, :int(w_lr / 256) * 256]
    h_lr, w_lr = height_field.shape[0], height_field.shape[1]

    offset_high = int(patch_size / 2)  # high res exemplar terrain #patch_size #
    print(offset_high)

    def get_sytle(index):
        width = int((w_lr - bound_w) / 2)
        index = int(index)
        if index < width:
            return [style_id_nn, style_id_nn], 0.5
        elif index > (width + bound_w):
            return [style_id_nn2, style_id_nn2], 0.5
        else:
            weight = 1.0 - (index - width) / bound_w
            return [style_id_nn, style_id_nn2], weight

    # process styles (high resolution terrain)
    patch_size_hr = patch_size * factor
    patch_mask_hr = build_patch(patch_size_hr)
    offset_high_hr = offset_high * factor

    # optimize the terrain with dictionary
    print("height_field.shape", height_field.shape)
    height_field = terrain_dilate(height_field, patch_size_hr)

    if smooth_input:
        T = butter2d(height_field, 0.008, 3)
    else:
        T = height_field

    h_T, w_T = T.shape
    d1 = math.floor((h_T - patch_size) / offset_high_hr)
    d2 = math.floor((w_T - patch_size) / offset_high_hr)
    dir_path = os.path.dirname(os.path.realpath(__file__)) + '/temp/'

    def rm_file(folder):
        for the_file in os.listdir(folder):
            file_path = os.path.join(folder, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)

    if os.path.isdir(dir_path):
        rm_file(dir_path + 'input')
        rm_file(dir_path + 'ref')
    else:
        os.mkdir(dir_path)
        os.mkdir(dir_path + 'input')
        os.mkdir(dir_path + 'ref')

    style_dic = {}
    weight_dic = {}
    for i in range(0, d1):
        for j in range(0, d2):
            if fix_ratio is None:
                style_dic[(i, j)], weight_dic[(i, j)] = get_sytle(
                    j * offset_high_hr + 0.5 * patch_size_hr - patch_size_hr)
            else:
                style_dic[(i, j)], weight_dic[(i, j)] = ([style_id_nn, style_id_nn2], fix_ratio)

    X, means_lr, current_r = build_dictionary_big(T, patch_mask_hr, offset_high_hr, d1, d2, 1,
                                                  save_dic=dir_path + 'input/', style_dic=style_dic,
                                                  weight_dic=weight_dic)

    a.output_dir = dir_path + 'result'
    a.input_dir = dir_path + 'input'
    a.test_dir = dir_path + 'input'
    a.ref_dir = dir_path + 'ref'
    a.mode = "test"
    a.png16bits = True
    a.checkpoint = os.path.dirname(os.path.realpath(__file__)) + '/checkpoint'

    if external_emb is None:
        inference_model(a)
    else:
        import shutil
        shutil.copy2(external_emb, dir_path + 'ref/style.png')
        emb = inference_emb(a)
        inference_model(a, emb)

    dir_path = os.path.dirname(os.path.realpath(__file__)) + '/temp/'

    result = np.zeros((h_lr + 2 * patch_size_hr, w_lr + 2 * patch_size_hr))
    for i in range(0, d1):
        for j in range(0, d2):
            style_id, style_id2 = style_dic[(i, j)]
            weight = weight_dic[(i, j)]
            dir_Y = dir_path + 'result/images/' + str(i) + '_' + str(j) + '_style_' + str(weight) + '_' + str(
                style_id2) + '_' + str(style_id) + '-outputs.png'
            image = cv2.imread(dir_Y, -1)
            image = np.double(image[:, :, 0])
            print(dir_Y, np.max(image), np.mean(image), np.min(image))
            image_r = (image - MAX_HEIGHT / 2) * current_r / (MAX_HEIGHT / 3)
            image_r = image_r * patch_mask_hr
            result[i * offset_high_hr: i * offset_high_hr + patch_size_hr,
            j * offset_high_hr:j * offset_high_hr + patch_size_hr] \
                += image_r + means_lr[i, j] * patch_mask_hr

    avg_mask = np.zeros(result.shape)
    for i in range(0, d1):
        for j in range(0, d2):
            avg_mask[i * offset_high_hr: i * offset_high_hr + patch_size_hr,
            j * offset_high_hr:j * offset_high_hr + patch_size_hr] \
                += patch_mask_hr
    nonzero_index = np.nonzero(avg_mask)

    result[nonzero_index] = np.divide(result[nonzero_index], avg_mask[nonzero_index])

    result = result[patch_size_hr: -patch_size_hr, patch_size_hr:-patch_size_hr]
    T = T[patch_size_hr: -patch_size_hr, patch_size_hr:-patch_size_hr]

    neg_index = np.where(result < 0)
    result[neg_index] = 0

    neg_index = np.where(T < 0)
    T[neg_index] = 0

    if not isUint16:
        result = result / 255 * MAX_HEIGHT
        T = T / 255 * MAX_HEIGHT
    result = np.floor(result)
    T = np.floor(T)

    exc_index = np.where(result >= MAX_HEIGHT)
    result[exc_index] = MAX_HEIGHT

    amplified_hf = result.astype('uint16')

    print('max height: ', np.amax(amplified_hf), '; min height: ', np.amin(amplified_hf))

    return amplified_hf, T.astype('uint16')


def generate_sample(input_filename, style_list, bound_w, resize, extra_label='', fix_ratio=None,
                    external_emb=None, smooth_input=True):
    """
        Function to generate samples, which will read the terrain, run the inference, and save the result
        :input: name:
                    filename for low resolution terrain
                style_list:
                    the style that need to transfer to the low-res terrain. This parameter can have the following options:
                        a). int: a int indicates the style id;
                        b). (int, int): a tuple of ints. It will generate terrain transists from one style to another
                        c). None: It means the reference style will be read from external_emb.
                bound_w:
                    when two styles are used, this parameter set the width of transist area.
                resize:
                    factor to resize the input.
                external_emb:
                    filename for out-style source
                smooth_input:
                    Whether smooth the input or not. If the input is low-res already, this should be False. If the input
                    has details already, set this parameter to True to get rid of its original details
        """

    hf = read_height_field(input_filename)
    name = input_filename.split('.')[-2][input_filename.rindex('/')+1:]
    if not bound_w is None:
        bound_w = int(bound_w * resize)
    hf = cv2.resize(hf, (int(resize * hf.shape[1]), int(resize * hf.shape[0])))
    amplified_hf, T = terrain_amplification_big(hf, 4, isUint16=True, style_list=style_list,
                                                bound_w=bound_w, fix_ratio=fix_ratio,
                                                external_emb=external_emb, smooth_input=smooth_input)
    if not style_list is None:
        try:
            style1, style2 = style_list
        except:
            style1 = style_list
            style2 = style_list
        style_label = str(style1) + '_' + str(style2)
    else:
        style_label = external_emb.split('/')[-1]
    cv2.imwrite('./inference/inference_sample/' + name + '_output_' + style_label + '_' + str(bound_w) + '_' + str(
        resize) + extra_label + '.png', amplified_hf)
    cv2.imwrite('./inference/inference_sample/' + name + '_input' + '_' + str(resize) + '.png', T)


if __name__ == '__main__':
    # Example: style interpolation (wouldn't run unless test data is avaiable).
    generate_sample(
        '/inference/infenrece_sample/input/your_own_test_sample.png',
        None, None, 1,
        extra_label='_test5', fix_ratio=1.0,
        external_emb='/inference/infenrece_sample/input/my_own_unseen_style_1_256_256.png',
        smooth_input=False)
