import numpy as np
import pdb

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def var_2d_mask(shape,acc):
    
    ''' producing variant 2D undersample mask with input shape and accelarate rate '''
    
    mask = np.random.binomial(1, np.ones(shape)*(1.0/acc))
    
    return np.float32(mask)

def down_sample_with_mask(ims):
    
    ''' simulate undersampled data with fully sampled image data and k-space sample mask '''
    
    # the acc is set to be 3 at this moment
    masks_single = var_2d_mask(shape=ims.shape[:-1],acc=3)
    
    # combine the 2 channels of ims into a complex 
    ims_cplx = ims[:,:,:,0] + ims[:,:,:,1]*1j
    
    ims_k_cplx = np.complex64(np.fft.fft2(ims_cplx)) # by default the last 2 axis
    
    k_sample_cplx = np.multiply(ims_k_cplx, masks_single)
    
    ims_sample_cplx = np.complex64(np.fft.ifft2(k_sample_cplx))
    
    ''' plot some figures '''
    # # plot some data to see results
    # for k in [1,2,3,10,100]:
    #     plt.figure()
    #     plt.subplot(1,3,1);plt.imshow(abs(ims_sample_cplx[k,:,:]));
    #     plt.subplot(1,3,2);plt.imshow(abs(ims_cplx[k,:,:]));
    #     plt.subplot(1,3,3);plt.imshow(masks_single[k,:,:]);
    #     plt.savefig('showims: '+str(k))
    #     plt.clf()
    
    ''' organize output '''
    # duplicate masks_single to have 2 channels
    masks = np.stack([masks_single, masks_single], axis=-1)
    
    ims_sample = np.stack([np.real(ims_sample_cplx), np.imag(ims_sample_cplx)], axis=-1)
    
    k_sample = np.stack([np.real(k_sample_cplx), np.imag(k_sample_cplx)], axis=-1)
    
    # return sampled recon images, and full k-space images
    return np.float32(ims_sample), np.float32(k_sample), np.float32(masks)
    
    