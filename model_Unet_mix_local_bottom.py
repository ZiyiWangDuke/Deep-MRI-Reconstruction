#!/usr/bin/env python
''' A improved model using Unet structure and mixed '''

from __future__ import print_function, division

import os, pdb
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import keras
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, Conv2DTranspose, Add, LocallyConnected2D
from .fft_layer import fft_layer
from .data_consistency_layer import data_consistency_with_mask_layer, symmetry_with_mask_layer
from .kspace_weight_layer import kspace_weight_layer, kspace_padding_layer

img_w = 128 # 256
img_h = 128 # 256

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

input_img = Input(( img_w, img_h, 2))
input_mask = Input(( img_w, img_h, 2))
input_k_sampled = Input(( img_w, img_h, 2)) # k space sampled
# input_img, input_mask, input_img_sampled = tf.unstack(inputs,axis=1)

# apply symetricity to the k-space data
refer = concatenate([input_mask, input_k_sampled], axis=-1)
# phase conjugate symmetry: skip this layer for complex input image

input_fft = fft_layer(fft_dir = True)(input_img)

# 5 conv sequential - using Segnet shape
conv1 = Conv2D(filters = 16, kernel_size = 3, strides=2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(input_fft)

conv1 = Conv2D(filters = 32, kernel_size = 3, strides=2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)

conv1 = kspace_padding_layer(pad_len=1)(conv1)

conv1 = LocallyConnected2D(filters = 8, kernel_size = 3, activation = 'relu', kernel_initializer = 'he_normal')(conv1)

conv1 = Conv2DTranspose(filters = 64, kernel_size = 3, strides=2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)

conv1 = Conv2DTranspose(filters = 32, kernel_size = 3, strides=2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)

conv1 = Conv2D(filters = 2, kernel_size = 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
# print("conv1 shape:",conv1.shape)

# residual
res1 = Add()([input_fft,conv1])
# add data consistency layer here

res1 = concatenate([res1, refer], axis=-1)
dc1 = data_consistency_with_mask_layer()(res1)

dc1 = fft_layer(fft_dir = False)(dc1)

# 5 conv sequential, stays in k-space
conv2 = Conv2D(filters = 16, kernel_size = 3, strides=2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dc1)

conv2 = Conv2D(filters = 32, kernel_size = 3, strides=2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)

conv2 = Conv2DTranspose(filters = 64, kernel_size = 3, strides=2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)

conv2 = Conv2DTranspose(filters = 32, kernel_size = 3, strides=2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)

conv2 = Conv2D(filters = 2, kernel_size = 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
# print("conv2 shape:",conv2.shape)

# residual
res2 = Add()([dc1,conv2])
# add data consistency layer here

res2 = fft_layer(fft_dir = True)(res2)

res2 = concatenate([res2, refer], axis=-1)
dc2 = data_consistency_with_mask_layer()(res2)


# 5 conv sequential, in Kspace 
conv3 = Conv2D(filters = 16, kernel_size = 3, strides=2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dc2)

conv3 = Conv2D(filters = 32, kernel_size = 3, strides=2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)

conv3 = Conv2DTranspose(filters = 64, kernel_size = 3, strides=2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)

conv3 = Conv2DTranspose(filters = 32, kernel_size = 3, strides=2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)

conv3 = Conv2D(filters = 2, kernel_size = 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
# print("conv3 shape:",conv3.shape)

# residual
res3 = Add()([dc2,conv3])
# add data consistency layer here

res3 = concatenate([res3, refer], axis=-1)
dc3 = data_consistency_with_mask_layer()(res3)

dc3 = fft_layer(fft_dir = False)(dc3)

# 5 conv sequential, in image domain
conv4 = Conv2D(filters = 16, kernel_size = 3, strides=2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dc3)

conv4 = Conv2D(filters = 32, kernel_size = 3, strides=2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)

conv4 = Conv2DTranspose(filters = 64, kernel_size = 3, strides=2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)

conv4 = Conv2DTranspose(filters = 32, kernel_size = 3, strides=2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)

conv4 = Conv2D(filters = 2, kernel_size = 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
# print("conv3 shape:",conv3.shape)

# residual
res4 = Add()([dc3,conv4])
# add data consistency layer here
# fft4 = fft_layer(fft_dir = True)(res4)

res4 = fft_layer(fft_dir = True)(res4)

res4 = concatenate([res4, refer], axis=-1)
dc4 = data_consistency_with_mask_layer()(res4)


# 5 conv sequential, in kspace
conv5 = Conv2D(filters = 16, kernel_size = 3, strides=2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dc4)

conv5 = Conv2D(filters = 32, kernel_size = 3, strides=2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)

conv5 = Conv2DTranspose(filters = 64, kernel_size = 3, strides=2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)

conv5 = Conv2DTranspose(filters = 32, kernel_size = 3, strides=2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)

conv5 = Conv2D(filters = 2, kernel_size = 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', 
               kernel_regularizer=keras.regularizers.l2(0.01),
               activity_regularizer=keras.regularizers.l1(0.01))(conv5)
# print("conv3 shape:",conv3.shape)

# residual
res5 = Add()([dc4,conv5])
# add data consistency layer here

# refer = symmetry_with_mask_layer()(refer)

res5 = concatenate([res5, refer], axis=-1)
dc5 = data_consistency_with_mask_layer()(res5)

dc5 = fft_layer(fft_dir = False)(dc5)

recon_encoder = Model(inputs = [input_img, input_mask, input_k_sampled], outputs = dc5)

# recon_encoder.summary()
# pdb.set_trace()
