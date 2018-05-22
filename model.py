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

# 5 conv sequential
conv4 = Conv2D(filters = 64, kernel_size = 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dc3)
print("conv3 shape:",conv3.shape)
conv4 = Conv2D(filters = 64, kernel_size = 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
print("conv3 shape:",conv3.shape)
conv4 = Conv2D(filters = 64, kernel_size = 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
print("conv3 shape:",conv3.shape)
conv4 = Conv2D(filters = 64, kernel_size = 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
print("conv3 shape:",conv3.shape)
conv4 = Conv2D(filters = 2, kernel_size = 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
print("conv3 shape:",conv3.shape)

# residual
res4 = Add()([dc3,conv4])
# add data consistency layer here
fft4 = fft_layer(fft_dir = True)(res4)

fft4 = concatenate([fft4, input_mask, input_img_sampled], axis=-1)
fft4 = data_consistency_with_mask_layer()(fft4)

dc4 = fft_layer(fft_dir = False)(fft4)

# 5 conv sequential
conv5 = Conv2D(filters = 64, kernel_size = 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(dc4)
print("conv3 shape:",conv3.shape)
conv5 = Conv2D(filters = 64, kernel_size = 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
print("conv3 shape:",conv3.shape)
conv5 = Conv2D(filters = 64, kernel_size = 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
print("conv3 shape:",conv3.shape)
conv5 = Conv2D(filters = 64, kernel_size = 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
print("conv3 shape:",conv3.shape)
conv5 = Conv2D(filters = 2, kernel_size = 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
print("conv3 shape:",conv3.shape)

# residual
res5 = Add()([dc4,conv5])
# add data consistency layer here
fft5 = fft_layer(fft_dir = True)(res5)

fft5 = concatenate([fft5, input_mask, input_img_sampled], axis=-1)
fft5 = data_consistency_with_mask_layer()(fft5)

dc5 = fft_layer(fft_dir = False)(fft5)

recon_encoder = Model(inputs = [input_img, input_mask, input_img_sampled], outputs = dc5)

pdb.set_trace()
