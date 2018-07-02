from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
import tensorflow as tf

class fft_layer(Layer):

    ''' FFT layer: compute fft for the input tensor'''

    def __init__(self, fft_dir = True, split='real_imag', **kwargs):
        # self.output_dim = output_dim
        self.fft_dir = fft_dir
        self.split = split # split = 'real_imag' or 'mag_phase'
        super(fft_layer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(fft_layer, self).build(input_shape)  # add method for build

    def call(self, x):
        # transfer float 32 [real, imaginary] to complex 64 for fft
        if self.split == 'real_imag':
            x_cplx = tf.complex(x[...,0],x[...,1])
        elif self.split == 'mag_phase': # mag_phase 
            x_cplx = tf.complex(tf.multiply(x[...,0],tf.cos(x[...,1])), tf.multiply(x[...,0],tf.sin(x[...,1])))
        else:
            raise Exception('un-recognizable split mode')   
            
        if self.fft_dir:
            x_cplx_fft = tf.fft2d(x_cplx)
        else:
            x_cplx_fft = tf.ifft2d(x_cplx)
        
        x_cplx_fft_4d = x_cplx_fft[..., None]
         
        if self.split == 'real_imag':
            # convert complex 64 [num, x, y] to float 32 [num, x, y, real/imag]
            x_fft = tf.concat([tf.real(x_cplx_fft_4d), tf.imag(x_cplx_fft_4d)], -1)
        elif self.split == 'mag_phase':
            # convert complex 64 [num, x, y] to float 32 [num, x, y, mag/phase]
            x_fft = tf.concat([tf.abs(x_cplx_fft_4d), tf.angle(x_cplx_fft_4d)], -1)
        else:
            raise Exception('un-recognizable split mode') 

        return(x_fft)

    def compute_output_shape(self, input_shape):
        return (input_shape)

class stack_layer(Layer):

    ''' stack layer: stack 2 tensors on the last axis'''

    def __init__(self, **kwargs):
        super(stack_layer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(stack_layer, self).build(input_shape)  # add method for build

    def call(self, x):
        # x with shape (,64,128) or x with shape (,128, 256)
        x_stack = tf.stack((x[:,:,:64],x[:,:,64:]),axis=-1)
        return(x_stack)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[1], 2)

if __name__ == "__main__":

    ''' test fft_layer '''
    from keras.layers import Input
    from keras.models import Model
    import pdb

    input = Input((256, 256, 2)) # do not consider batch as the first dimension

    mid = fft_layer(fft_dir=True,split='mag_phase')(input)
    # output = fft_layer(fft_dir=False,split='mag_phase')(mid)

    model = Model(inputs=input, outputs=mid)

    data_in = np.ones((2,256,256,2))
    data_in[:,:,:,1] = np.pi/2

    data_out = model.predict(data_in)
    pdb.set_trace()
