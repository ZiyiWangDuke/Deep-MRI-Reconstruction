from dl_research.common.data import (
    DataPipeline,
    image_transforms,
    transforms,
    volume_transforms,
)

from dl_research.projects.degrade.scripts.slices import (
    MCD1_METADATA_CSV,
    MCD2_METADATA_CSV,
)
from dl_research.projects.quality_ae_spatial import data_transforms
from dl_research.papers.automap.transforms import PullSliceKey, to_freq_space
from dl_research.papers.automap.data import axial_non_dw_filter_256, load_data

import pdb,os
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# undersample in k-space, pre-organize the data for training
def to_kspace_undersample(*, image_key='images', kspace_key = 'kspace', under_image_key= 'under_sample_image', mask_key='sample_mask', k_sample_key='k_samples'):
    
    def transform_img(img):
        acc = 3
        shape = img.shape[:2]
        
        im_k_space = np.fft.fft2(np.squeeze(img, axis=-1))
        # pdb.set_trace()
        im_k_space = np.stack([np.real(im_k_space), np.imag(im_k_space)], axis=-1)
        
        # generate random sample mask
        mask = np.random.binomial(1, np.ones(shape)*(1.0/acc))
        mask = np.stack([mask,mask],axis=-1)
        
        k_sample = np.multiply(im_k_space, mask)
        
        under_image = np.fft.ifft2(k_sample[:,:,0]+k_sample[:,:,1]*1j)
        under_image = np.stack([np.real(under_image), np.imag(under_image)], axis=-1)
        
        # change here
        return im_k_space, mask, k_sample, under_image

    def transform(data):
        big_lst  = [
            transform_img(img)
            for img in data[image_key]
        ]
        
        im_k_space = [element[0] for element in big_lst]
        mask = [element[1] for element in big_lst]
        k_sample = [element[2] for element in big_lst]
        under_image = [element[3] for element in big_lst]

        return {**data, kspace_key: im_k_space, mask_key: mask, k_sample_key: k_sample, under_image_key: under_image}

    return transform

# create data pipeline using Michal's code
def create_data_pipeline_eval(batch_size):
    return DataPipeline(
        transforms.make_batches(batch_size),
        transforms.transform_batches(
            transforms.load_batch_data(
                volume_transforms.load_volumes(),
            ),
            data_transforms.RandomSampleSlicesValid(axis = 2),
            PullSliceKey(axis = 2),
            to_kspace_undersample(),
            image_transforms.set_data_format('channel_last'),
        ),
        transforms.buffer_data(),
    )

def get_data_gen_model(data_type, batch_size):
    
    # data generator directory for training the model
    # data_type: train or valid
    batch_iterator = create_data_pipeline_eval(batch_size=batch_size)(data_type)
    
    while True:
        dic_data = next(batch_iterator)
        
        # get list from dictionary and convert to array
        ims_sample = np.stack(dic_data["under_sample_image"], axis=0)
        masks = np.stack(dic_data["sample_mask"], axis=0)
        k_sample = np.stack(dic_data["k_samples"], axis=0)
        
        yield [ims_sample, masks, k_sample]
    

if __name__ == '__main__':
    
    data_train, data_valid = load_data(
    data_csvs=[(MCD1_METADATA_CSV, os.path.dirname(MCD1_METADATA_CSV))], df_filter=axial_non_dw_filter_256)
    
    batch_iterator_train = get_data_gen_model(data_type=data_train, batch_size=32)
    batch_iterator_valid = get_data_gen_model(data_type=data_valid, batch_size=32)
    
    arr = next(batch_iterator_train)
    
    ims_sample = arr[0]
    masks = arr[1]
    k_sample = arr[2]
    
    for k in range(10):
        # pdb.set_trace()
        plt_mask = abs(masks[-k,:,:,0])
        plt_im_sample = abs(ims_sample[-k,:,:,0]+ims_sample[-k,:,:,1]*1j)
        plt_k_sample = abs(k_sample[-k,:,:,0]+k_sample[-k,:,:,1]*1j)

        plt.figure()
        plt.subplot(1,4,1); plt.imshow(plt_mask)
        plt.subplot(1,4,2); plt.imshow(plt_im_sample)
        plt.subplot(1,4,3); plt.imshow(plt_k_sample)
        plt.savefig('output/figures/predict_im_'+str(k))
        plt.clf()
    
    pdb.set_trace()