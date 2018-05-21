from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
import tensorflow as tf

class fft_layer(Layer):

    ''' FFT layer: compute fft for the input tensor'''

    def __init__(self, fft_dir = True, **kwargs):
        # self.output_dim = output_dim
        self.fft_dir = fft_dir
        super(fft_layer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(fft_layer, self).build(input_shape)  # add method for build

    def call(self, x):
        # transfer float 32 [real, imaginary] to complex 64 for fft
        x_cplx = tf.complex(x[...,0],x[...,1])

        if self.fft_dir:
            x_cplx_fft = tf.fft2d(x_cplx)
        else:
            x_cplx_fft = tf.ifft2d(x_cplx)

        # convert complex 64 [num, x, y] to float 32 [num, x, y, real/imag]
        x_cplx_fft_4d = x_cplx_fft[..., None]
        x_fft = tf.concat([tf.real(x_cplx_fft_4d), tf.imag(x_cplx_fft_4d)], -1)

        return(x_fft)

    def compute_output_shape(self, input_shape):
        return (input_shape)


if __name__ == "__main__":

    ''' test fft_layer '''
    from keras.layers import Input
    from keras.models import Model
    import pdb

    input = Input((256, 256, 2)) # do not consider batch as the first dimension

    mid = fft_layer(fft_dir=True)(input)
    # output = fft_layer(fft_dir=False)(mid)

    model = Model(inputs=input, outputs=mid)

    data_in = np.ones((2,256,256,2))/100
    data_in[1,:] = 0*data_in[1,:]

    data_out = model.predict(data_in)
    pdb.set_trace()
