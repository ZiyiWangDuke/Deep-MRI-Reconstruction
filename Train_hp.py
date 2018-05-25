from dl_research.projects.degrade.scripts.slices import (
    MCD1_METADATA_CSV,
    MCD2_METADATA_CSV,
)

from dl_research.papers.automap.data import axial_non_dw_filter_256, load_data

from .utils_functions import var_2d_mask, down_sample_with_mask, relative_error_center_30pix
from .model_Unet import recon_encoder

import os, pdb
import numpy as np
import nibabel as nib
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import keras 



project_path = '/home/zwang/dl_research/projects/under_recon/'

# function read in all data as a big dataframe 
data_train, data_valid = load_data(
    data_csvs=[(MCD1_METADATA_CSV, os.path.dirname(MCD1_METADATA_CSV))],
    df_filter=axial_non_dw_filter_256,
)

# take a small bite of the big data set as the trainning set
num_list = 60 #for training

img_w = 256
img_h = 256
img_c = 2

small_list = data_train.iloc[:num_list,:].filekey
ims = np.array([]).reshape(0,img_w,img_h)

del data_train
del data_valid

for index in range(num_list):
    nift_im = nib.load(small_list[index])
    nift_im = nift_im.get_data()
    nift_im = nift_im.transpose([2,0,1]) # slice, x, y
    
    ims = np.concatenate([ims, nift_im], axis=0)
    
    print("finished attach subject "+str(index))
    
# attach a empty imag channel to the image stack

# shuffle the ims, along the first axis
np.random.shuffle(ims)

# normalize slice 
ims = np.float32(ims)
# num_s = ims.shape[0]
# ims = (ims-np.mean(ims,axis=(1,2)).reshape((num_s,1,1)))/np.std(ims,axis=(1,2)).reshape((num_s,1,1))

# add complex axis
ims = np.expand_dims(ims,axis=-1)
ims = np.concatenate([ims, np.zeros(ims.shape)],axis=-1)

# simulate downsampled ims with 2D undersample mask
ims_sample, masks, k_sample = down_sample_with_mask(ims)

# # test kspace symetry layer
# from keras.layers import Input
# from keras.models import Model
# from .data_consistency_layer import symmetry_with_mask_layer

# input = Input((256, 256, 4)) # do not consider batch as the first dimension
# output = symmetry_with_mask_layer()(input)

# model = Model(inputs=input, outputs=output)

# input_data = np.concatenate((masks, k_sample),axis=-1)
# output_ims = model.predict(input_data)
# # output_ims = output_ims[:,:,:,2] + 1j*output_ims[:,:,:,3] # fetch image data from output
# # output_ims = np.fft.fft2(output_ims)

# for k in range(10):
#     plt_mask = abs(masks[-k,:,:,0])
#     plt_im_sample = abs(ims_sample[-k,:,:,0]+ims_sample[-k,:,:,1]*1j)
#     plt_im_predict = abs(output_ims[-k,:,:])
#     plt_full = abs(ims[-k,:,:,0]+ims[-k,:,:,1]*1j)
    
#     plt.figure()
#     plt.subplot(1,4,1); plt.imshow(plt_mask)
#     plt.subplot(1,4,2); plt.imshow(plt_im_sample)
#     plt.subplot(1,4,3); plt.imshow(plt_im_predict)
#     plt.subplot(1,4,4); plt.imshow(plt_full)
#     plt.savefig('output/figures/phase_sym_'+str(k))
    
# pdb.set_trace()
# build image data generator to send input 

batch_size = 10
epochs = 20

''' load model and train '''
tensorboard = keras.callbacks.TensorBoard(log_dir="output/keras_logs/{}".format(datetime.strftime(datetime.now(), '%Y_%m_%d_%H_%M')),
                                          write_grads=False, # write grads take a significant amount of memory
                                          write_images=False,
                                          histogram_freq=2)

# the last slices to be used for test, exclude from training
num_ck = 10

input_data = [ims_sample[:-num_ck,:,:,:], masks[:-num_ck,:,:,:], k_sample[:-num_ck,:,:,:]]
output_data = ims[:-num_ck,:,:,:]


recon_encoder.summary()
recon_encoder.compile(loss="mean_squared_error", optimizer='adam', metrics=[relative_error_center_30pix],sample_weight_mode='temporal')

history = recon_encoder.fit(  
                      x=input_data,
                      y=output_data,
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=1,
                      callbacks=[tensorboard],
                      validation_split=0.1,
                      shuffle=True,
                      initial_epoch=0)

recon_encoder.save_weights('output/models/under_recon_180524_sym.h5')

# recon_encoder.load_weights('output/models/under_recon_180523.h5')

for k in range(num_ck):
    # pdb.set_trace()
    test_input = [ims_sample[-k,:,:,:], masks[-k,:,:,:], k_sample[-k,:,:,:]]
    test_input = [np.expand_dims(x, axis=0) for x in test_input] # make a dummy first dimension c
    
    model_out = recon_encoder.predict(test_input)
    
    plt_mask = abs(masks[-k,:,:,0])
    plt_im_sample = abs(ims_sample[-k,:,:,0]+ims_sample[-k,:,:,1]*1j)
    plt_im_predict = abs(model_out[0,:,:,0]+model_out[0,:,:,1]*1j)
    plt_full = abs(ims[-k,:,:,0]+ims[-k,:,:,1]*1j)
    
    plt.figure()
    plt.subplot(1,4,1); plt.imshow(plt_mask)
    plt.subplot(1,4,2); plt.imshow(plt_im_sample)
    plt.subplot(1,4,3); plt.imshow(plt_im_predict)
    plt.subplot(1,4,4); plt.imshow(plt_full)
    plt.savefig('output/figures/predict_im_'+str(k))
    plt.clf()

# check data consistency layer
# dc1 = recon_encoder.layers[11]
# dc2 = recon_encoder.layers[21]
# dc3 = recon_encoder.layers[31]
# dc4 = recon_encoder.layers[41]

# shuffle
pdb.set_trace()


