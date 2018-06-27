# recon the undersampled data acquired from the scanner, June 26. 2018

import numpy as np
import nibabel as nib
from scipy import interpolate

import pdb

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import keras
from .model_Unet import recon_encoder
from .utils_functions import down_sample_with_mask
import time

ims_sample = np.load('scanner_data_0626/under_sample/img_volume.npy')
masks = np.load('scanner_data_0626/under_sample/mask_volume.npy')
k_sample = np.load('scanner_data_0626/under_sample/k_volume.npy')

# img_out = np.zeros(img_128.shape)
# zero_filling = np.zeros(img_128.shape)
# img_full = np.zeros(img_128.shape)

# model import
print('start loading model')
# recon_encoder.load_weights('output/models/under_recon_180606_unet_mixlocal_big.h5') 
recon_encoder.load_weights('output/models/under_recon_180606_unet_aug.h5')
print('finished loading model')

# seperate complex images to real and imaginary
ims_sample = ims_sample.astype('complex64')
ims_sample = np.stack((np.real(ims_sample),np.imag(ims_sample)),axis=-1)

k_sample = k_sample.astype('complex64')
k_sample = np.fft.fftshift(k_sample,axes=(1,2))
k_sample = np.stack((np.real(k_sample),np.imag(k_sample)),axis=-1)

masks = masks.astype('float32')
masks = np.fft.fftshift(masks,axes=(1,2))
masks = np.stack((masks,masks),axis=-1)

# predict with the model
start_time = time.time()
model_out = recon_encoder.predict([ims_sample, masks, k_sample])
end_time = time.time()
print(end_time-start_time)
# model_out = ims_sample

model_out = model_out[:,:,:,0]+model_out[:,:,:,1]*1j

ims_sample = ims_sample[:,:,:,0]+ims_sample[:,:,:,1]*1j

print('finished processing all images, average processing time')
for k in range(ims_sample.shape[0]):
    
    plt_mask = abs(masks[k,:,:,0])
    plt_im_sample = abs(ims_sample[k,:,:])
    plt_im_predict = abs(model_out[k,:,:])
    
    plt_u_phase = np.angle(ims_sample[k,:,:])
    plt_c_phase = np.angle(model_out[k,:,:])
    vmin = np.min(plt_u_phase)
    vmax = np.max(plt_c_phase)
    
    # plt_full = abs(input_ims[k,:,:,0]+input_ims[k,:,:,1]*1j)
    
    plt.figure()
    # plt.subplot(1,4,1); plt.imshow(np.fft.fftshift(plt_mask)); plt.title('sample');plt.axis('off')
    plt.subplot(1,4,1); plt.imshow(plt_im_sample);plt.title('Under Sample');plt.axis('off')
    plt.subplot(1,4,3); plt.imshow(plt_im_predict);plt.title('CNN Recon');plt.axis('off')
    plt.subplot(1,4,2); plt.imshow(plt_u_phase,vmin=vmin,vmax=vmax);plt.title('U Phase');plt.axis('off')
    plt.subplot(1,4,4); plt.imshow(plt_c_phase,vmin=vmin,vmax=vmax);plt.title('C Phase');plt.axis('off')
    plt.savefig('output/figures/comp_'+str(k))
    plt.close()
    print('printing figure '+str(k))

pdb.set_trace()

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

        plt.savefig('output/figures/predict_hp_unet_300_aug_mix_'+str(m))
        plt.close()


        print('printing figure '+str(m))