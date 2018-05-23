import numpy as np
import pdb
import keras
import keras.backend as K
import tensorflow as tf

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
    return np.float32(ims_sample), np.float32(masks), np.float32(k_sample)


## keras call back function, observe output per epoch
class OutputObserver(keras.callbacks.Callback):
    
    """ callback to observe the output of the after each epoch """
    
    def __init__(self, test_downsample, test_fullsample):
        # pdb.set_trace()
        self.test_downsample = test_downsample
        self.test_fullsample = abs(test_fullsample[:,:,:,0] + test_fullsample[:,:,:,1]*1j)
        self.num, self.im_w, self.im_h = test_fullsample.shape[:3]

    def on_epoch_end(self, epoch, logs={}):
        pdb.set_trace()
        model_out = self.model.predict(self.test_downsample)
        
        model_out = abs(model_out[:,:,:,0] + model_out[:,:,:,1]*1j)
        
        plt_down = model_out.reshape(self.num*self.im_w,self.im_h)
        plt_full = self.test_fullsample.reshape(self.num*self.im_w,self.im_h)

        plt_block = np.concatenate((plt_down,plt_full),axis=1)
        plt.imshow(plt_block)
        plt.savefig('output/figures/epoch_out_at_'+str(epoch))
        # pdb.set_trace()


        

def relative_error_center_30pix(y_true, y_pred):
    # calculate the average of relative error of the center 30 pixels
    
    # assume it is a square image
    dim = 256
    center = 30
    dim_st = np.int32(dim/2-center)
    dim_end = np.int32(dim/2+center)

    # combine last 2 channels into complex number
    y_true_cplx = tf.complex(y_true[:,:,:,0],y_true[:,:,:,1])
    y_pred_cplx = tf.complex(y_pred[:,:,:,0],y_pred[:,:,:,1])
    
    # crop out the center part (FOV)
    ct_y_true = tf.abs(y_true_cplx[:,dim_st:dim_end,dim_st:dim_end])
    ct_y_pred = tf.abs(y_pred_cplx[:,dim_st:dim_end,dim_st:dim_end])
    ct_dif = tf.subtract(ct_y_true, ct_y_pred)
    
    ave_error_per = tf.div(tf.reduce_mean(ct_dif), tf.reduce_mean(ct_y_true))
    return ave_error_per


# def data_gen(uteList,sourceDir):
#     list_len = len(uteList)
#     index = 0

#     while True:
        
#     # pdb.set_trace()
#     yield curImg, curMask, curWeight

    