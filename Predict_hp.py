''' use the model to predict HP data'''
from dl_research.projects.degrade.scripts.slices import (
    MCD1_METADATA_CSV,
    MCD2_METADATA_CSV,
)

from dl_research.papers.automap.data import axial_non_dw_filter_256, load_data
from dl_research.common.data.hri import load_split_data_from_dicom_conversion

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
        get_anatomical_filter(df) &
        get_t2_filter(df) &
        (df.nifti_shape.str[0] == 256) &
        (df.nifti_shape.str[1] == 256)
    )

from .utils_functions import var_2d_mask, down_sample_with_mask, relative_error_center_30pix, loss_weight_mse
from .model import recon_encoder
from .data_gen import get_data_gen_model

import os, pdb
import numpy as np
import nibabel as nib
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import keras 

recon_encoder.load_weights('output/models/under_recon_180530_original.h5')
data_csv = '/data2/hyperfine/dl-data-annotator/degrade/2018-03-06-1/data.csv'

data_train, data_valid = load_split_data_from_dicom_conversion(
    data_csvs=[(data_csv, os.path.dirname(data_csv))],
    # data_csvs=[(MCD1_METADATA_CSV, os.path.dirname(MCD1_METADATA_CSV))],
    # df_filter=zwang_filter,
    
)

batch_size = 10
epochs = 20
steps_per_epoch = int(len(data_train)/batch_size)
# validation_steps = int(len(data_valid)/batch_size)
epoch_size = batch_size*steps_per_epoch

batch_iterator_train = get_data_gen_model(data_type=data_train, batch_size=batch_size, epoch_size=epoch_size, flag='train')
# batch_iterator_valid = get_data_gen_model(data_type=data_valid, batch_size=batch_size, epoch_size=epoch_size, flag='validation')


# ims = np.float32(ims)
# # num_s = ims.shape[0]
# # ims = (ims-np.mean(ims,axis=(1,2)).reshape((num_s,1,1)))/np.std(ims,axis=(1,2)).reshape((num_s,1,1))

# # add complex axis
# ims = np.expand_dims(ims,axis=-1)
# ims = np.concatenate([ims, np.zeros(ims.shape)],axis=-1)

# # simulate downsampled ims with 2D undersample mask
# ims_sample, masks, k_sample = down_sample_with_mask(ims)
test_input = next(batch_iterator_train)
# test_input = next(batch_iterator_train)
# pdb.set_trace()
model_out = recon_encoder.predict(test_input[0]) # input

masks = test_input[0][1]
ims = test_input[1]
ims_sample = test_input[0][0]

# pdb.set_trace()

for k in range(batch_size):
    # pdb.set_trace()
    # test_input = [ims_sample[-k,:,:,:], masks[-k,:,:,:], k_sample[-k,:,:,:]]
    # test_input = [np.expand_dims(x, axis=0) for x in test_input] # make a dummy first dimension 

    
    plt_mask = abs(masks[k,:,:,0])
    plt_im_sample = abs(ims_sample[k,:,:,0]+ims_sample[k,:,:,1]*1j)
    plt_im_predict = abs(model_out[k,:,:,0]+model_out[k,:,:,1]*1j)
    plt_full = abs(ims[k,:,:,0]+ims[k,:,:,1]*1j)
    
    plt.figure()
    plt.subplot(1,4,1); plt.imshow(plt_mask)
    plt.subplot(1,4,2); plt.imshow(plt_im_sample)
    plt.subplot(1,4,3); plt.imshow(plt_im_predict)
    plt.subplot(1,4,4); plt.imshow(plt_full)
    plt.savefig('output/figures/predict_train_'+str(k))
    plt.close()

