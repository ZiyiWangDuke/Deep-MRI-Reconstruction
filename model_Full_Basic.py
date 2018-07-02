#!/usr/bin/env python
''' A improved model using Unet structure '''

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
from keras.layers import Input, concatenate, Conv2D, Conv2DTranspose, Add, BatchNormalization, Activation, MaxPooling2D, UpSampling2D, Dense, Reshape, Permute,Dropout
from .fft_layer import fft_layer, stack_layer
from .data_consistency_layer import data_consistency_with_mask_layer, symmetry_with_mask_layer

img_w = 128 #256
img_h = 128 #256
n_sample = 5670 # number of kspace sampling acquired

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# input_img = Input(( img_w, img_h, 2))
# input_mask = Input(( img_w, img_h, 2))
input_k_sampled_real = Input(( 1,n_sample )) # k space sampled real
input_k_sampled_imag = Input(( 1,n_sample )) # k space sampled real

full_real = Dense(64*64, input_shape=(n_sample,))(input_k_sampled_real)
full_real = Dropout(rate=0.5)(full_real)

full_imag = Dense(64*64, input_shape=(n_sample,))(input_k_sampled_imag)
full_imag = Dropout(rate=0.5)(full_imag)

full_real = Reshape((64, 64))(full_real)
full_imag = Reshape((64, 64))(full_imag)

conv0 = concatenate([full_real, full_imag], axis=-1)
conv0 = stack_layer()(conv0)

# conv0 = Conv2DTranspose(filters = 64, kernel_size = 3, strides=2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv0)
conv0 = Conv2D(filters = 64, kernel_size = 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv0)

conv0 = Conv2DTranspose(filters = 2, kernel_size = 3, strides=2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv0)

conv0 = fft_layer(fft_dir = False)(conv0)
# input_img, input_mask, input_img_sampled = tf.unstack(inputs,axis=1)

# apply symetricity to the k-space data
# refer = concatenate([input_mask, input_k_sampled], axis=-1)
# phase conjugate symmetry: skip this layer for complex input image

# 5 conv sequential - using Segnet shape
# pdb.set_trace()

conv1 = Conv2D(filters = 32, kernel_size = 3, strides=2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv0)

conv1 = Conv2D(filters = 64, kernel_size = 3, strides=2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)

conv1 = Conv2DTranspose(filters = 128, kernel_size = 3, strides=2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)

conv1 = Conv2DTranspose(filters = 64, kernel_size = 3, strides=2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)

conv1 = Conv2D(filters = 2, kernel_size = 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
# print("conv1 shape:",conv1.shape)

# residual
res1 = Add()([conv0,conv1])
# add data consistency layer here
# fft1 = fft_layer(fft_dir = True, split='mag_phase')(res1)

# fft1 = concatenate([fft1, refer], axis=-1)
# fft1 = data_consistency_with_mask_layer()(fft1)

# dc1 = fft_layer(fft_dir = False)(fft1)

# 5 conv sequential
conv2 = Conv2D(filters = 16, kernel_size = 3, strides=2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(res1)

conv2 = Conv2D(filters = 32, kernel_size = 3, strides=2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)

conv2 = Conv2DTranspose(filters = 64, kernel_size = 3, strides=2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)

conv2 = Conv2DTranspose(filters = 32, kernel_size = 3, strides=2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)

conv2 = Conv2D(filters = 2, kernel_size = 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
# print("conv2 shape:",conv2.shape)

# residual
res2 = Add()([res1,conv2])
# add data consistency layer here
# fft2 = fft_layer(fft_dir = True, split='mag_phase')(res2)

# fft2 = concatenate([fft2, refer], axis=-1)
# fft2 = data_consistency_with_mask_layer()(fft2)

# dc2 = fft_layer(fft_dir = False)(fft2)

# 5 conv sequential
conv3 = Conv2D(filters = 32, kernel_size = 3, strides=2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(res2)

conv3 = Conv2D(filters = 64, kernel_size = 3, strides=2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)

conv3 = Conv2DTranspose(filters = 128, kernel_size = 3, strides=2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)

conv3 = Conv2DTranspose(filters = 64, kernel_size = 3, strides=2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)

conv3 = Conv2D(filters = 2, kernel_size = 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
# print("conv3 shape:",conv3.shape)

# residual
res3 = Add()([res2,conv3])
# add data consistency layer here
# fft3 = fft_layer(fft_dir = True, split='mag_phase')(res3)

# fft3 = concatenate([fft3, refer], axis=-1)
# fft3 = data_consistency_with_mask_layer()(fft3)

# dc3 = fft_layer(fft_dir = False)(fft3)

# 5 conv sequential
conv4 = Conv2D(filters = 16, kernel_size = 3, strides=2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(res3)

conv4 = Conv2D(filters = 32, kernel_size = 3, strides=2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)

conv4 = Conv2DTranspose(filters = 64, kernel_size = 3, strides=2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)

conv4 = Conv2DTranspose(filters = 32, kernel_size = 3, strides=2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)

conv4 = Conv2D(filters = 2, kernel_size = 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
# print("conv3 shape:",conv3.shape)

# residual
res4 = Add()([res3,conv4])
# add data consistency layer here
# fft4 = fft_layer(fft_dir = True, split='mag_phase')(res4)

# fft4 = concatenate([fft4, refer], axis=-1)
# fft4 = data_consistency_with_mask_layer()(fft4)

# dc4 = fft_layer(fft_dir = False)(fft4)

# 5 conv sequential
conv5 = Conv2D(filters = 32, kernel_size = 3, strides=2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(res4)

conv5 = Conv2D(filters = 16, kernel_size = 3, strides=2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)

conv5 = Conv2DTranspose(filters = 128, kernel_size = 3, strides=2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)

conv5 = Conv2DTranspose(filters = 64, kernel_size = 3, strides=2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)

conv5 = Conv2D(filters = 2, kernel_size = 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', 
               kernel_regularizer=keras.regularizers.l2(0.01),
               activity_regularizer=keras.regularizers.l1(0.01))(conv5)
# print("conv3 shape:",conv3.shape)

# residual
res5 = Add()([res4,conv5])
# add data consistency layer here
# fft5 = fft_layer(fft_dir = True, split='mag_phase')(res5)

# refer = symmetry_with_mask_layer()(refer)

# fft5 = concatenate([fft5, refer], axis=-1)
# fft5 = data_consistency_with_mask_layer()(fft5)

# dc5 = fft_layer(fft_dir = False)(fft5)

recon_encoder = Model(inputs = [input_k_sampled_real, input_k_sampled_imag], outputs = res5)

# pdb.set_trace()
