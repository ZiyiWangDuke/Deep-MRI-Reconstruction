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

# first fill the image to 128*128
img_refill = np.zeros((128,128,36))
img_refill[11:117,:,:] = img

x = np.linspace(-128, 128, 128)
y = np.linspace(-128, 128, 128)

xn = np.linspace(-128, 128, 256)
yn = np.linspace(-128, 128, 256)

img_256 = np.zeros((256,256,img.shape[2]))

# interpolate from 128*128 to 256*256
for k in range(img.shape[2]):
    f = interpolate.interp2d(x=x, y=y, z=img_refill[:,:,k], kind='cubic')
    img_256[:,:,k] = f(xn,yn)

img_256 = np.transpose(img_256,(2,0,1))
img_256 = np.stack((img_256,np.zeros(img_256.shape)),axis=-1)

ims_sample, masks, k_sample, down_rate = down_sample_with_mask(img_256)

# model import
recon_encoder.load_weights('output/models/under_recon_180530_unet.h5')
# pdb.set_trace()

model_out = recon_encoder.predict([ims_sample, masks, k_sample])

for k in range(img_256.shape[0]):
    
    plt_mask = abs(masks[k,:,:,0])
    plt_im_sample = abs(ims_sample[k,:,:,0]+ims_sample[k,:,:,1]*1j)
    plt_im_predict = abs(model_out[k,:,:,0]+model_out[k,:,:,1]*1j)
    plt_full = abs(img_256[k,:,:,0]+img_256[k,:,:,1]*1j)
    
    plt.figure()
    plt.subplot(1,4,1); plt.imshow(np.fft.fftshift(plt_mask)); plt.title('sample {:.2f}'.format(down_rate))
    plt.subplot(1,4,2); plt.imshow(plt_im_sample);plt.title('zero_filling')
    plt.subplot(1,4,3); plt.imshow(plt_im_predict);plt.title('CNN recon')
    plt.subplot(1,4,4); plt.imshow(plt_full);plt.title('Full sample')
    plt.savefig('output/figures/predict_hp_unet_using_'+str(k))
    plt.close()
    print('printing figure '+str(k))
    