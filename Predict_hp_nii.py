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

img = nib.load('dl_research/projects/under_recon/hp_data_1.nii')
img= img.get_data()

img_refill = img[:,11:117,:]

# first fill the image to 128*128
# img_refill = np.zeros((128,128,36))
# img_refill[11:117,:,:] = img

x = np.linspace(-64, 64, 106)
y = np.linspace(-64, 64, 106)

xn = np.linspace(-64, 64, 128)
yn = np.linspace(-64, 64, 128)

img_256 = np.zeros((128,128,img.shape[2]))

# interpolate from 128*128 to 256*256
for k in range(img.shape[2]):
    f = interpolate.interp2d(x=x, y=y, z=img_refill[:,:,k], kind='cubic')
    img_256[:,:,k] = f(xn,yn)

img_256 = np.transpose(img_256,(2,0,1))
img_256 = np.stack((img_256,np.zeros(img_256.shape)),axis=-1)

ims_sample, masks, k_sample, down_rate = down_sample_with_mask(img_256)

# k_sample=np.roll(k_sample, 1, axis=1)
# k_sample=np.roll(k_sample, 1, axis=2)
# model import

recon_encoder.load_weights('output/models/under_recon_180605_unet_distmesh_200.h5')
# pdb.set_trace()

import time

start_time = time.time()

# jiggle the mask, for CNN
# mask_slice = masks[:,:,:,0]
# for c in range(masks.shape[0]):
#     for k in range(127):
#         for r in range(127):
#             mask_slice[c,k,r] = mask_slice[c,k+1,r+1]
# masks = np.stack([mask_slice,mask_slice],axis=-1)

model_out = recon_encoder.predict([ims_sample, masks, k_sample])
end_time = time.time()
print((end_time-start_time)/model_out.shape[0])

# jiggle the ksample, for zerofilling
# for c in range(k_sample.shape[0]):
#     for k in range(127):
#         for r in range(127):
#             k_sample[c,k,r,:] = k_sample[c,k+1,r+1,:]


for k in range(img_256.shape[0]):
    
    plt_mask = abs(masks[k,:,:,0])
    plt_im_sample = abs(np.fft.ifft2(k_sample[k,:,:,0]+k_sample[k,:,:,1]*1j))
    plt_im_predict = abs(model_out[k,:,:,0]+model_out[k,:,:,1]*1j)
    plt_full = abs(img_256[k,:,:,0]+img_256[k,:,:,1]*1j)
    
    
    plt.figure(figsize=(10, 8))
    
    plt.subplot(2,4,1); plt.imshow(np.fft.fftshift(plt_mask),interpolation='none'); plt.title('sample {:.2f}'.format(down_rate));plt.axis('off')
    plt.subplot(2,4,2); plt.imshow(plt_im_sample);plt.title('zero_filling');plt.axis('off')
    plt.subplot(2,4,3); plt.imshow(plt_im_predict);plt.title('CNN recon');plt.axis('off')
    plt.subplot(2,4,4); plt.imshow(plt_full);plt.title('Full sample');plt.axis('off')
    
    dif1 = abs(plt_im_sample-plt_full)
    dif2 = abs(plt_im_predict-plt_full)

    mean_dif1 = np.mean(dif1)/(np.mean(plt_full)*1.0)
    mean_dif2 = np.mean(dif2)/(np.mean(plt_full)*1.0)
    vmin = np.min(dif1)
    vmax = np.max(dif2)
    
    plt.subplot(2,4,6); plt.imshow(dif1,vmin=vmin,vmax=vmax);plt.title('R_dif {:.2f}'.format(mean_dif1));plt.axis('off')
    plt.subplot(2,4,7); plt.imshow(dif2,vmin=vmin,vmax=vmax);plt.title('R_dif {:.2f}'.format(mean_dif2));plt.axis('off')
    plt.tight_layout()
    plt.savefig('output/figures/predict_hp_unet_test_jig'+str(k))
    plt.close()
    print('printing figure '+str(k))
    