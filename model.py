#!/usr/bin/env python
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
from keras.layers import Input, concatenate, Conv2D, Add
from fft_layer import fft_layer
from data_consistency_layer import data_consistency_with_mask_layer

img_w = 256
img_h = 256

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

input_img = Input(( img_w, img_h, 2))
input_mask = Input(( img_w, img_h, 2))
input_img_sampled = Input(( img_w, img_h, 2))
# input_img, input_mask, input_img_sampled = tf.unstack(inputs,axis=1)

# 5 conv sequential
conv1 = Conv2D(filters = 64, kernel_size = 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(input_img)
print("conv1 shape:",conv1.shape)
conv1 = Conv2D(filters = 64, kernel_size = 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
print("conv1 shape:",conv1.shape)
conv1 = Conv2D(filters = 64, kernel_size = 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
print("conv1 shape:",conv1.shape)
conv1 = Conv2D(filters = 64, kernel_size = 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
print("conv1 shape:",conv1.shape)
conv1 = Conv2D(filters = 2, kernel_size = 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
print("conv1 shape:",conv1.shape)

# residual
res1 = Add()([input_img,conv1])
# add data consistency layer here
fft1 = fft_layer(fft_dir = True)(res1)

fft1 = concatenate([fft1, input_mask, input_img_sampled], axis=-1)
fft1 = data_consistency_with_mask_layer()(fft1)

dc1 = fft_layer(fft_dir = False)(fft1)

# 5 conv sequential
conv2 = Conv2D(filters = 64, kernel_size = 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dc1)
print("conv2 shape:",conv2.shape)
conv2 = Conv2D(filters = 64, kernel_size = 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
print("conv2 shape:",conv2.shape)
conv2 = Conv2D(filters = 64, kernel_size = 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
print("conv2 shape:",conv2.shape)
conv2 = Conv2D(filters = 64, kernel_size = 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
print("conv2 shape:",conv2.shape)
conv2 = Conv2D(filters = 2, kernel_size = 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
print("conv2 shape:",conv2.shape)

# residual
res2 = Add()([dc1,conv2])
# add data consistency layer here
fft2 = fft_layer(fft_dir = True)(res2)

fft2 = concatenate([fft2, input_mask, input_img_sampled], axis=-1)
fft2 = data_consistency_with_mask_layer()(fft2)

dc2 = fft_layer(fft_dir = False)(fft2)

# 5 conv sequential
conv3 = Conv2D(filters = 64, kernel_size = 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dc2)
print("conv3 shape:",conv3.shape)
conv3 = Conv2D(filters = 64, kernel_size = 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
print("conv3 shape:",conv3.shape)
conv3 = Conv2D(filters = 64, kernel_size = 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
print("conv3 shape:",conv3.shape)
conv3 = Conv2D(filters = 64, kernel_size = 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
print("conv3 shape:",conv3.shape)
conv3 = Conv2D(filters = 2, kernel_size = 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
print("conv3 shape:",conv3.shape)

# residual
res3 = Add()([dc2,conv3])
# add data consistency layer here
fft3 = fft_layer(fft_dir = True)(res3)

fft3 = concatenate([fft3, input_mask, input_img_sampled], axis=-1)
fft3 = data_consistency_with_mask_layer()(fft3)

dc3 = fft_layer(fft_dir = False)(fft3)

autoencoder = Model(inputs = [input_img, input_mask, input_img_sampled], outputs = dc3)
opt = keras.optimizers.adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.00001)
autoencoder.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy'],sample_weight_mode='temporal')
pdb.set_trace()
