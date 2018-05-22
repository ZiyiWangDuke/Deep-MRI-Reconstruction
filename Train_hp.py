from dl_research.projects.degrade.scripts.slices import (
    MCD1_METADATA_CSV,
    MCD2_METADATA_CSV,
)

from dl_research.papers.automap.data import axial_non_dw_filter_256, load_data

from .utils_functions import var_2d_mask, down_sample_with_mask
from .model import recon_encoder

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
num_list = 10

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
ims = np.expand_dims(ims,axis=-1)
ims = np.concatenate([ims, np.zeros(ims.shape)],axis=-1)
ims = np.float32(ims)

# shuffle the ims, along the first axis
np.random.shuffle(ims)

# simulate downsampled ims with 2D undersample mask
ims_sample, k_sample, masks = down_sample_with_mask(ims)

input_data = [ims_sample, masks, k_sample]
output_data = ims

# pdb.set_trace()
# build image data generator to send input 

batch_size = 2
epochs = 10

''' load model and train '''
# tensorboard = keras.callbacks.TensorBoard(log_dir="keras_logs/{}".format(datetime.strftime(datetime.now(), '%Y_%m_%d_%H')),
#                                           write_grads=False,
#                                           write_images=False,
#                                           histogram_freq=0)

recon_encoder.summary()
recon_encoder.compile(loss="mean_squared_error", optimizer='adam', metrics=['accuracy'],sample_weight_mode='temporal')

history = recon_encoder.fit(  x=input_data,
                      y=output_data,
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=1,
                      # callbacks=[tensorboard],
                      validation_split=0.2,
                      shuffle=True,
                      initial_epoch=0)

recon_encoder.save('under_recon_180522.h5')


# shuffle
pdb.set_trace()
    




