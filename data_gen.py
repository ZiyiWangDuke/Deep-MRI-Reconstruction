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

import scipy.stats as st

def jiggle_mask(mask):
    
    ''' jiggle the mask by at most 2 pixel '''
    # return coordinate
    dlen = mask.shape[1]
    x, y = np.nonzero(mask)
    
    x = x+np.random.randint(low=-2, high=2, size=x.shape)
    y = y+np.random.randint(low=-2, high=2, size=y.shape)

    x[x>=dlen] = dlen-1; x[x<0] = 0
    y[y>=dlen] = dlen-1; y[y<0] = 0    
    
    mask_jig =np.zeros(mask.shape)
    mask_jig[x,y] = 1
    
    return mask_jig

def read_2D_Distmesh():
    
    ''' read in 2D distmesh previously generated '''
    ind = np.random.randint(0,20)
    
    mask_file = 'mask_{}.npy'.format(ind)
    mask = np.load('dl_research/projects/under_recon/dist_masks/'+mask_file)
    
    return mask

def gen_2D_Distmesh(dlen=128):
    
    ''' generate distmesh with varied density, idealy for 128*128 points '''
    
    import distmesh as dm

    
    ind = random.uniform(0.25, 0.35) # down-sample rate from 0.38 to 0.29
    
    fd = lambda p: dm.ddiff(dm.drectangle(p,-1,1,-1,1),dm.dcircle(p,0,0,0.01))
    fh = lambda p: 0.2+ind*dm.dcircle(p,0,0,0.01) # increase the second para decrease the sample rate

    # 0.3 x 128 points -- 30% undersampling

    pt, tttt = dm.distmesh2d(fd, fh, 0.03, (-1,-1,1,1),[(-1,-1),(-1,1),(1,-1),(1,1)])

    # mapp = np.zeros((dlen+1,dlen+1))

    x = np.array([int(np.round(k)) for k in pt[:,0]*dlen/2+dlen/2])
    y = np.array([int(np.round(k)) for k in pt[:,1]*dlen/2+dlen/2])

    x[x>(dlen-1)] = dlen-1
    y[y>(dlen-1)] = dlen-1

    mask = np.zeros((dlen,dlen))
    mask[x,y] = 1
    
    return mask
    
def gen_2D_Gaussian(kernlen, nsig):
    
    ''' Returns a 2D Gaussian kernel array '''
    # kernlen: length of mask, in our case 256
    # nsig: like SD, 2.2: 50%, 3.5: 25%
    
    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    
    kernel = kernel_raw/np.max(kernel_raw)

    return kernel

def to_kspace_undersample(*, image_key='images', kspace_key = 'kspace', under_image_key= 'under_sample_image', mask_key='sample_mask', k_sample_key='k_samples'):
    
    '''undersample in k-space, pre-organize the data for training'''
    
    def transform_img(img):
        # downsample from 256 to 128
        img=img[::2,::2,:]
        
        shape = img.shape[:2]
        traj_key = 'distmesh' 
        
        
        im_k_space = np.fft.fft2(np.squeeze(img, axis=-1))
        # pdb.set_trace()
        im_k_space = np.stack([np.real(im_k_space), np.imag(im_k_space)], axis=-1)
        
        if traj_key == 'random':
            # generate random sample mask
            acc = 3
            mask = np.random.binomial(1, np.ones(shape)*(1.0/acc))
        elif traj_key == 'gaussian':
            # generate random gaussian sample mask
            kernel = gen_2D_Gaussian(kernlen=shape[1], nsig=2.95) # acc = 3
            mask = np.random.binomial(1, np.multiply(np.ones(shape),kernel))
            mask = np.fft.fftshift(mask)
        elif traj_key == 'distmesh':
            mask = read_2D_Distmesh()
            mask = np.fft.fftshift(mask)
        else:
            raise ValueError('A bad traj_key is set, check data_gen')
        
        # mask_jig = jiggle_mask(mask)
        
        # mask_jig = np.stack([mask_jig,mask_jig],axis=-1)
        mask = np.stack([mask,mask],axis=-1)
        # mask: 128*128*2
        
        k_sample = np.multiply(im_k_space, mask)
        
        under_image = np.fft.ifft2(k_sample[:,:,0]+k_sample[:,:,1]*1j)
        under_image = np.stack([np.real(under_image), np.imag(under_image)], axis=-1)
        
        # change here
        return im_k_space, mask, k_sample, under_image, img

    def transform(data):
        big_lst  = [
            transform_img(img)
            for img in data[image_key]
        ]
        
        im_k_space = [element[0] for element in big_lst]
        mask = [element[1] for element in big_lst]
        k_sample = [element[2] for element in big_lst]
        under_image = [element[3] for element in big_lst]
        
        # if we do downsampling to 128
        image_128 = [element[4] for element in big_lst]

        return {**data, kspace_key: im_k_space, mask_key: mask, k_sample_key: k_sample, under_image_key: under_image, image_key:image_128}

    return transform

# create data pipeline using Michal's code
def create_data_pipeline_eval(batch_size, epoch_size, flag):
    scale_limits = (0.78, 1.20)
    rotation_limits = (-5., 5.)
    translation_limits = (-10., 10.)
    snr_db = 9.0
    data_shape = (256,256)
    
    if flag == 'train':
        return DataPipeline(
            transforms.sample_dataframe_by_subject(epoch_size), # number of samples in each epoch
            transforms.make_batches(batch_size),
            transforms.transform_batches(
                transforms.load_batch_data(
                    volume_transforms.load_volumes(),
                ),
                # gen transform
                data_transforms.generate_slices_similarity_transforms(
                    target_size_in_plane=data_shape,
                    scale=scale_limits,
                    rotation=rotation_limits,
                    translation=translation_limits,
                ),
                # apply transform
                volume_transforms.similarity_transform_volumes(
                    data_shape,
                    target_shapes_key='target_shapes',
                ),
                # add noise
                data_transforms.AddNoise(snr_db=snr_db),
                
                data_transforms.RandomSampleSlices(axis = 2), # RandomSampleSlicesValid gives the same images
                PullSliceKey(axis = 2),
                to_kspace_undersample(),
                image_transforms.set_data_format('channel_last'),
            ),
            transforms.buffer_data(),
        )
    elif flag == 'validation':
        return DataPipeline(
            transforms.weight_dataframe_by_inverse_subject_frequency(), # number of samples in each epoch
            transforms.make_batches(batch_size),
            transforms.transform_batches(
                transforms.load_batch_data(
                    volume_transforms.load_volumes(),
                ),
                
                # use sudo-random, always the same noise 
                data_transforms.AddNoiseValid(snr_db=snr_db),
                
                data_transforms.RandomSampleSlicesValid(axis = 2), # RandomSampleSlicesValid gives the same images
                PullSliceKey(axis = 2),
                to_kspace_undersample(),
                image_transforms.set_data_format('channel_last'),
            ),
            transforms.buffer_data(),
        )

def get_data_gen_model(data_type, batch_size, flag, epoch_size=0):
    
    # data generator directory for training the model
    # data_type: train or valid
    batch_iterator = create_data_pipeline_eval(batch_size=batch_size, epoch_size=epoch_size, flag=flag)(data_type)
        
    while True:
        # dirty way of starting it over from the beginning
        try:
            dic_data = next(batch_iterator)
        except:
            batch_iterator = create_data_pipeline_eval(batch_size=batch_size, epoch_size=epoch_size, flag=flag)(data_type)
            dic_data = next(batch_iterator)
               
        # get list from dictionary and convert to array
        ims_sample = np.float32(np.stack(dic_data["under_sample_image"], axis=0))
        masks = np.float32(np.stack(dic_data["sample_mask"], axis=0))
        k_sample = np.float32(np.stack(dic_data["k_samples"], axis=0))
        
        # return original image as well
        ims = np.float32(np.stack(dic_data["images"], axis=0))
        # add an extra dim for ims
        ims = np.concatenate([ims, np.zeros(ims.shape)], axis=-1)
        
        yield [ims_sample, masks, k_sample], ims
    

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