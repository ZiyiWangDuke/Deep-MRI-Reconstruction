import numpy as np
import nibabel as nib
from scipy import interpolate

import pdb

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import keras
from .model_Unet_top import recon_encoder
from .utils_functions import down_sample_with_mask
import time

img_128 = np.load('dl_research/projects/under_recon/single_channel_pad_noise.npy')
# img_128 = np.log(abs(img_128))
# img_128 = np.log(np.abs(img_128))
# shape (8, 7, 128, 128, 36)
# normalize the image slice wise
thre = np.percentile(img_128,95.0,axis=(2,3))
thre = np.expand_dims(thre,axis=2)
thre = np.expand_dims(thre,axis=2)
thre = np.tile(thre, (1,1,128,128,1))

img_128 = img_128/thre
img_128[img_128>1] = 1

## rotate
img_128 = np.rot90(img_128, k=-1, axes=(2,3))


img_out = np.zeros(img_128.shape)
zero_filling = np.zeros(img_128.shape)
img_full = np.zeros(img_128.shape)

# model import
print('start loading model')
# recon_encoder.load_weights('output/models/under_recon_180606_unet_mixlocal_big.h5') 
recon_encoder.load_weights('output/models/under_recon_180607_unet_top.h5')
print('finished loading model')

key = 'comb_contrast_coil'
start_time = time.time()
# key = 'comb_contrast'

for k in range(img_128.shape[0]): # iterate through coils
    for m in range(img_128.shape[1]): # iterate through contrast

        input_ims = img_128[k,m,:,:,:]

        input_ims = np.transpose(input_ims,(2,0,1))
        input_ims = np.stack((input_ims,np.zeros(input_ims.shape)),axis=-1)

        ims_sample, masks, k_sample, down_rate = down_sample_with_mask(input_ims, flag='distmesh')

        model_out = recon_encoder.predict([ims_sample, masks, k_sample])
        
        model_out = abs(model_out[:,:,:,0]+model_out[:,:,:,1]*1j)
        img_out[k,m,:,:,:] = np.transpose(model_out,(1,2,0))
        
        ims_sample = abs(ims_sample[:,:,:,0]+ims_sample[:,:,:,1]*1j)
        zero_filling[k,m,:,:,:] = np.transpose(abs(ims_sample),(1,2,0))
        
        input_ims = abs(input_ims[:,:,:,0]+input_ims[:,:,:,1]*1j)
        img_full[k,m,:,:,:] = np.transpose(abs(input_ims),(1,2,0))

end_time = time.time()
print('finished processing all images, average processing time')
print((end_time-start_time)/(8.0*7.0*36.0))
# for k in range(input_ims.shape[0]):
    
#     plt_mask = abs(masks[k,:,:,0])
#     plt_im_sample = abs(ims_sample[k,:,:,0]+ims_sample[k,:,:,1]*1j)
#     plt_im_predict = abs(model_out[k,:,:,0]+model_out[k,:,:,1]*1j)
#     plt_full = abs(input_ims[k,:,:,0]+input_ims[k,:,:,1]*1j)
    
#     plt.figure()
#     plt.subplot(1,4,1); plt.imshow(np.fft.fftshift(plt_mask)); plt.title('sample {:.2f}'.format(down_rate));plt.axis('off')
#     plt.subplot(1,4,2); plt.imshow(plt_im_sample);plt.title('zero_filling');plt.axis('off')
#     plt.subplot(1,4,3); plt.imshow(plt_im_predict);plt.title('CNN recon');plt.axis('off')
#     plt.subplot(1,4,4); plt.imshow(plt_full);plt.title('Full sample');plt.axis('off')
#     plt.savefig('output/figures/predict_hp_unet_single_'+str(k))
#     plt.close()
#     print('printing figure '+str(k))


#### single coil-single slice
if key == 'comb_contrast':
    for k in range(img_128.shape[0]):
        # coil 
        for m in range(img_128.shape[4]):
        # location

            plt_mask = abs(masks[0,:,:,0])
            plt_im_sample = np.sum(zero_filling[k,:,:,:,m], axis=0)
            plt_im_predict = np.sum(img_out[k,:,:,:,m], axis=0)
            plt_full = np.sum(img_full[k,:,:,:,m], axis=0)

            plt.figure()

            vmin = np.min(plt_full)
            vmax = np.max(plt_full)

            plt.subplot(2,4,1); plt.imshow(np.fft.fftshift(plt_mask)); plt.title('sample {:.2f}'.format(down_rate));plt.axis('off')
            plt.subplot(2,4,2); plt.imshow(plt_im_sample,vmin=vmin,vmax=vmax);plt.title('zero_filling');plt.axis('off')
            plt.subplot(2,4,3); plt.imshow(plt_im_predict,vmin=vmin,vmax=vmax);plt.title('CNN recon');plt.axis('off')
            plt.subplot(2,4,4); plt.imshow(plt_full,vmin=vmin,vmax=vmax);plt.title('Full sample');plt.axis('off')

            dif1 = abs(plt_im_sample-plt_full)
            dif2 = abs(plt_im_predict-plt_full)

            mean_dif1 = np.mean(dif1)/(np.mean(plt_full)*1.0)
            mean_dif2 = np.mean(dif2)/(np.mean(plt_full)*1.0)

            vmin = np.min(dif1)
            vmax = np.max(dif2)

            plt.subplot(2,4,6); plt.imshow(dif1,vmin=vmin,vmax=vmax);plt.title('{:.4f}'.format(mean_dif1));plt.axis('off')
            plt.subplot(2,4,7); plt.imshow(dif2,vmin=vmin,vmax=vmax);plt.title('{:.4f}'.format(mean_dif2));plt.axis('off')

            plt.savefig('output/figures/predict_hp_unet_single_coil_'+str(k)+'_slice_'+str(m))
            plt.close()


            print('printing figure '+str(k))
        
## combined coils with the dumbest square of sum method
if key == 'comb_contrast_coil':
    plt_im_sample_c = np.zeros((img_128.shape[0],img_128.shape[2],img_128.shape[3]))
    plt_im_predict_c = np.zeros((img_128.shape[0],img_128.shape[2],img_128.shape[3]))
    plt_full_c = np.zeros((img_128.shape[0],img_128.shape[2],img_128.shape[3]))

    for m in range(img_128.shape[4]):
        # slice
        for k in range(img_128.shape[0]):
        # coil

            plt_im_sample_c[k,:,:] = np.sum(zero_filling[k,:,:,:,m], axis=0)
            plt_im_predict_c[k,:,:] = np.sum(img_out[k,:,:,:,m], axis=0)
            plt_full_c[k,:,:] = np.sum(img_full[k,:,:,:,m], axis=0)

        plt_mask = abs(masks[0,:,:,0])
        plt_im_sample = np.sum(np.square(plt_im_sample_c), axis=0)
        plt_im_predict = np.sum(np.square(plt_im_predict_c), axis=0)
        plt_full = np.sum(np.square(plt_full_c), axis=0)

        plt.figure()

        vmin = np.min(plt_full)
        vmax = np.max(plt_full)*1.0/2.0
        
        # pdb.set_trace()

        plt.subplot(2,4,1); plt.imshow(np.fft.fftshift(plt_mask)); plt.title('sample {:.2f}'.format(down_rate));plt.axis('off')
        plt.subplot(2,4,2); plt.imshow(plt_im_sample,vmin=vmin,vmax=vmax);plt.title('zero_filling');plt.axis('off')
        plt.subplot(2,4,3); plt.imshow(plt_im_predict,vmin=vmin,vmax=vmax);plt.title('CNN recon');plt.axis('off')
        plt.subplot(2,4,4); plt.imshow(plt_full,vmin=vmin,vmax=vmax);plt.title('Full sample');plt.axis('off')

        dif1 = abs(plt_im_sample-plt_full)
        dif2 = abs(plt_im_predict-plt_full)

        mean_dif1 = np.mean(dif1)/(np.mean(plt_full)*1.0)
        mean_dif2 = np.mean(dif2)/(np.mean(plt_full)*1.0)

        vmin = np.min(dif1)
        vmax = np.max(dif1)*1.0/3.0
        
        plt.subplot(2,4,6); plt.imshow(dif1,vmin=vmin,vmax=vmax);plt.title('{:.4f}'.format(mean_dif1));plt.axis('off')
        plt.subplot(2,4,7); plt.imshow(dif2,vmin=vmin,vmax=vmax);plt.title('{:.4f}'.format(mean_dif2));plt.axis('off')
        
        ## plot k-space
        ft_dif1 = abs(abs(np.fft.fftshift(np.fft.ifft2(plt_im_sample)) - abs(np.fft.fftshift(np.fft.ifft2(plt_full)))))
        ft_dif2 = abs(abs(np.fft.fftshift(np.fft.ifft2(plt_im_predict)) - abs(np.fft.fftshift(np.fft.ifft2(plt_full)))))
        
        vmin = np.min(ft_dif1)
        vmax = np.max(ft_dif1)*1.0/4.0
        
        plt.subplot(2,4,5); plt.imshow(ft_dif1,vmin=vmin,vmax=vmax);plt.axis('off')
        plt.subplot(2,4,8); plt.imshow(ft_dif2,vmin=vmin,vmax=vmax);plt.axis('off')

        plt.savefig('output/figures/predict_hp_unet_'+str(m))
        plt.close()


        print('printing figure '+str(m))