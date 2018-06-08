import numpy as np
import pdb
import keras
import keras.backend as K
import tensorflow as tf

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import nibabel as nib

import scipy.io as sio
import scipy.stats as st

from .data_gen import read_2D_Distmesh

def gen_2D_Gaussian(kernlen, nsig):
    
    """Returns a 2D Gaussian kernel array."""
    # kernlen: length of mask, in our case 256
    # nsig: like SD, 2.2: 50%, 3.5: 25%
    
    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    
    kernel = kernel_raw/np.max(kernel_raw)

    return kernel

def read_int_mask(discreet_len, depth, file_flag):
    
    ''' read in mask from mat, and extrapolate it into 256*256 '''
    
    # read in coordinate from .mat
    if(file_flag == 'strange'):
        # the strange shape
        img = sio.loadmat('dl_research/projects/under_recon/kcoords_1.mat')
        cords = img['s']
        cords = np.transpose(cords,(1,0))
        # shape 2*num
    elif(file_flag == 'using'):
        # the big spiral circle that misses the outpart
        import h5py
        arrays = {}
        f = h5py.File('dl_research/projects/under_recon/kcoords.mat')
        for k, v in f.items():
            arrays[k] = np.array(v)

        cords = arrays['kcoords']
    
    mask = np.zeros((discreet_len,discreet_len))

    for k in range(cords.shape[1]):

        x = int(np.round(cords[0,k]*discreet_len))
        y = int(np.round(cords[1,k]*discreet_len))
        mask[x,y] = 1
    
    undersample_rate = np.sum(mask)/(1.0*discreet_len*discreet_len)
    # stack mask into the shape required
    mask = np.expand_dims(mask,axis=0)
    mask = np.tile(mask,(depth,1,1))
    
    # return mask size depth*img_w*img_h
    return np.float32(mask), undersample_rate
        
def var_2d_mask(shape,acc):
    
    ''' producing variant 2D undersample mask with input shape and accelarate rate '''
    
    mask = np.random.binomial(1, np.ones(shape)*(1.0/acc))
    
    # shape: (36, 256, 256)
    # center = 1
    mask[:,98:158,98:158] = 1
    
    undersample_rate = np.sum(mask)/(np.prod(mask.shape)*1.0)
    
    return np.float32(mask), undersample_rate

def down_sample_with_mask(ims, flag):
    
    ''' simulate undersampled data with fully sampled image data and k-space sample mask '''
    
    # flag = 'distmesh'
    
    # the acc is set to be 3 at this moment
    if flag == 'random':
        masks_single, undersample_rate = var_2d_mask(shape=ims.shape[:-1],acc=3)
        # shape 36*256*256
        
    elif flag == 'gaussian':
        kernel = gen_2D_Gaussian(kernlen=ims.shape[1], nsig=2.95) # acc = 3, 2.95
        masks_single = np.random.binomial(1, np.multiply(np.ones(ims.shape[1]),kernel))
        masks_single = np.fft.fftshift(masks_single)
        
        undersample_rate = np.sum(masks_single)/np.prod(masks_single.shape)
        
        masks_single = np.expand_dims(masks_single,axis=0)
        masks_single = np.tile(masks_single,(ims.shape[0],1,1))
    elif flag == 'distmesh':
        masks_single = read_2D_Distmesh()
        masks_single = np.fft.fftshift(masks_single)
        
        undersample_rate = np.sum(masks_single)/np.prod(masks_single.shape)
        
        masks_single = np.expand_dims(masks_single,axis=0)
        masks_single = np.tile(masks_single,(ims.shape[0],1,1))
        # shape 128*128
    else:
        masks_single, undersample_rate = read_int_mask(discreet_len=ims.shape[1], depth=ims.shape[0], file_flag=flag)
    
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
    return np.float32(ims_sample), np.float32(masks), np.float32(k_sample), undersample_rate


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
    
    ''' Keras Metric: calculate the average of relative error of the center 30 pixels '''
    
    # assume it is a square image
    dim = 128
    center = 30
    dim_st = np.int32(dim/2-center)
    dim_end = np.int32(dim/2+center)

    # combine last 2 channels into complex number
    y_true_cplx = tf.complex(y_true[:,:,:,0],y_true[:,:,:,1])
    y_pred_cplx = tf.complex(y_pred[:,:,:,0],y_pred[:,:,:,1])
    
    # crop out the center part (FOV)
    ct_y_true = tf.abs(y_true_cplx[:,dim_st:dim_end,dim_st:dim_end])
    ct_y_pred = tf.abs(y_pred_cplx[:,dim_st:dim_end,dim_st:dim_end])
    ct_dif = tf.abs(tf.subtract(ct_y_true, ct_y_pred))
    
    ave_error_per = tf.div(tf.reduce_mean(ct_dif), tf.reduce_mean(ct_y_true))
    return ave_error_per

def psnr_tensor(y_true, y_pred):
    
    ''' Keras Metric: PSNR of float32'''
    
    return tf.image.psnr(y_true, y_pred, max_val=1.0)

def ssim_tensor(y_true, y_pred):
    
    ''' Keras Metric: SSIM of float32'''
    
    return tf.image.ssim(y_true, y_pred, max_val=1.0)

def loss_weight_mse(y_true, y_pred):
    
    ''' Custom loss function, weight by value '''
    
    mse_w = K.sum(K.square(y_pred - y_true)*K.abs(y_true))
    
    return mse_w

# def data_gen(uteList,sourceDir):
#     list_len = len(uteList)
#     index = 0

#     while True:
        
#     # pdb.set_trace()
#     yield curImg, curMask, curWeight

    