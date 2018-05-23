from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
import tensorflow as tf

class data_consistency_with_mask_layer(Layer):

    ''' Data consistency layer with mask'''

    def __init__(self, inv_noise_level=None, **kwargs):
        super(data_consistency_with_mask_layer, self).__init__(**kwargs)
        self.inv_noise_level = inv_noise_level

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        # self.lam = self.add_weight(name='lambda',
        #                            initializer='uniform',
        #                            # initializer = keras.initializers.Constant(value=100.0),
        #                            shape=(),
        #                            trainable=True)

        super(data_consistency_with_mask_layer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs, **kwargs):

        '''
        Inputs: 3x4d tensors: k-space from image(in pipeline), mask, k-space samples

        output: 4d tensor, input with entries replaced/weighted with the sampled values
        '''

        x = inputs[...,0:2]
        mask = inputs[...,2:4]
        xk_sampled = inputs[...,4:6]

        # if self.lam:  # noisy case
        #     output = (x + self.lam * xk_sampled) / (1 + self.lam)
        # else:  # noiseless case, essentially just using the original data
        #     output = (1 - mask) * x + xk_sampled
        output = (1 - mask) * x + xk_sampled

        return output

    def compute_output_shape(self, input_shapes):

        # shape from (?,256,256,6) to (?,256,256,2)
        output_shape = (input_shapes[0], input_shapes[1], input_shapes[2], 2)
        return output_shape # last dimension comes from stacking img, mask and img_sampled

if __name__ == "__main__":

    ''' test data_consistency layer '''

    from keras.layers import Input, Conv2D, concatenate
    from keras.models import Model
    import tensorflow as tf
    import pdb

    input1 = Input((256, 256, 2)) # do not consider batch as the first dimension
    input2 = Input((256, 256, 2))
    input3 = Input((256, 256, 2))

    input = concatenate([input1,input2,input3], axis=-1)
    output = data_consistency_with_mask_layer()(input)
    output = Conv2D(filters = 2, kernel_size = 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(output)
    # output = fft_layer(fft_dir=False)(mid)

    model = Model(inputs=[input1, input2, input3], outputs=output)

    data_in = np.ones((10,256,256,2))
    data_in = np.stack([data_in, data_in, data_in], axis=-1)
    # data_in[1,:] = 0*data_in[1,:]
    pdb.set_trace()
    data_out = model.predict(data_in)
    pdb.set_trace()
