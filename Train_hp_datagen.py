# from dl_research.projects.degrade.scripts.slices import (
#     MCD1_METADATA_CSV,
#     MCD2_METADATA_CSV,
# )

# RUN_LABEL="unet_phase_ir" USERNAME=zwang ./qsub-run.sh  python -m dl_research.projects.under_recon.Train_hp_datagen

from dl_research.projects.degrade.scripts.slices import (
    get_anatomical_filter,
    get_axial_filter,
    get_non_diffusion_filter,
    get_t2_filter,
    MCD1_METADATA_CSV,
)

def zwang_filter(df):
    return (
        get_axial_filter(df) &
        get_non_diffusion_filter(df) &
        get_t2_filter(df) &
        get_anatomical_filter(df) &
        (df.nifti_shape.str[0] == 256) &
        (df.nifti_shape.str[1] == 256)
    )

from dl_research.common.data.hri import load_split_data_from_dicom_conversion

from .utils_functions import var_2d_mask, down_sample_with_mask, relative_error_center_30pix, loss_weight_mse
from .utils_keras import TensorBoardImage
from .model_Unet import recon_encoder
from .data_gen import get_data_gen_model

import os, pdb
import numpy as np
import nibabel as nib
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import keras 

# project_path = '/home/zwang/dl_research/projects/under_recon/'

data_train, data_valid = load_split_data_from_dicom_conversion(
    data_csvs=[(MCD1_METADATA_CSV, os.path.dirname(MCD1_METADATA_CSV))],
    df_filter=zwang_filter,
    valid_frac = 0.2 
)

batch_size = 10
epochs = 30
slice_subject = 22 # approximate

steps_per_epoch = int(len(data_train)*slice_subject/batch_size/10)
validation_steps = int(len(data_valid)*slice_subject/batch_size/5)

epoch_size = steps_per_epoch * batch_size

batch_iterator_train = get_data_gen_model(data_type=data_train, batch_size=batch_size, epoch_size=epoch_size, flag='train')
batch_iterator_valid = get_data_gen_model(data_type=data_valid, batch_size=batch_size, epoch_size=epoch_size, flag='validation')

# [ims_sample, masks, k_sample], ims = next(batch_iterator_train)
# for k in range(10):
#     plt.imshow(ims[k,:,:,0]);plt.axis('off')
#     plt.savefig('output/figures/test_gen'+str(k))

# pdb.set_trace()

del data_train
del data_valid

''' callback functions to monitor trainning progress '''

tensor_log_dir = "output/keras_logs/{}_unet_phase_unet_r_i/".format(datetime.strftime(datetime.now(), '%Y_%m_%d_%H_%M'))

tensorboard = keras.callbacks.TensorBoard(log_dir=tensor_log_dir,
                                          write_grads=False, # write grads take a significant amount of memory
                                          write_images=False,
                                          histogram_freq=0)

# tensorimages = TensorBoardImage(log_dir=tensor_log_dir, tag='Predicted Imags', num_ims = 5, test_ims=next(batch_iterator_valid)[0])

''' load model and train '''

recon_encoder.summary()
recon_encoder.compile(loss="mean_squared_error", optimizer='adam', metrics=[relative_error_center_30pix],sample_weight_mode='temporal')

history = recon_encoder.fit_generator(
                      generator=batch_iterator_train, 
                      steps_per_epoch=steps_per_epoch, 
                      epochs=epochs, 
                      callbacks=[tensorboard], 
                      validation_data=batch_iterator_valid, 
                      validation_steps=validation_steps)

recon_encoder.save_weights('output/models/under_recon_180627_unet_phase_unet_r_i.h5')

# recon_encoder.load_weights('output/models/under_recon_180523.h5')

# for k in range(num_ck):
#     # pdb.set_trace()
#     test_input = [ims_sample[-k,:,:,:], masks[-k,:,:,:], k_sample[-k,:,:,:]]
#     test_input = [np.expand_dims(x, axis=0) for x in test_input] # make a dummy first dimension c
    
#     model_out = recon_encoder.predict(test_input)
    
#     plt_mask = abs(masks[-k,:,:,0])
#     plt_im_sample = abs(ims_sample[-k,:,:,0]+ims_sample[-k,:,:,1]*1j)
#     plt_im_predict = abs(model_out[0,:,:,0]+model_out[0,:,:,1]*1j)
#     plt_full = abs(ims[-k,:,:,0]+ims[-k,:,:,1]*1j)
    
#     plt.figure()
#     plt.subplot(1,4,1); plt.imshow(plt_mask)
#     plt.subplot(1,4,2); plt.imshow(plt_im_sample)
#     plt.subplot(1,4,3); plt.imshow(plt_im_predict)
#     plt.subplot(1,4,4); plt.imshow(plt_full)
#     plt.savefig('output/figures/predict_im_'+str(k))
#     plt.clf()

# check data consistency layer
# dc1 = recon_encoder.layers[11]
# dc2 = recon_encoder.layers[21]
# dc3 = recon_encoder.layers[31]
# dc4 = recon_encoder.layers[41]

# shuffle
pdb.set_trace()


