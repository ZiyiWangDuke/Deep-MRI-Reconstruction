from keras import backend as K
import keras
import pdb
from keras.engine.topology import Layer
import numpy as np
import tensorflow as tf

class symmetry_with_mask_layer(Layer):
    
    ''' K-space symetricity layer with mask '''
    # input: concatenate mask and xk_sample
    # output: updated mask and xk_sample with k-space symetricity
    # confidence value assigned to the new mask
    
    def __init__(self, **kwargs):
        super(symmetry_with_mask_layer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        # Create a trainable weight variable for this layer, confidence of the symetry 
        self.confidence = self.add_weight(name='confidence',
                                   # initializer='uniform',
                                   initializer = keras.initializers.Constant(1.0),
                                   shape=(),
                                   trainable=True)
        
        super(symmetry_with_mask_layer, self).build(input_shape)
        
    def call(self, inputs, **kwargs):
        
        # This is pre-fftshift, so the symetry is:
        # x, y not 0: x+y=256, or x+y=257
        # x or y ==0: only odd number has sym  
        
        num_pix = inputs.shape[1]
        
        mask = inputs[...,0:2]
        xk_sampled = inputs[...,2:4]
        
        # remove row 0 and col 0, because no fftshift was found 
        mask_headless = mask[:,1:,1:,:]
        xk_sampled_headless = xk_sampled[:,1:,1:,:]
        
        mask_inv = tf.subtract(tf.constant(1,dtype=tf.float32),mask_headless) # inverse mask tensor
        mask_flip = tf.image.flip_left_right(tf.image.flip_up_down(mask_headless))
        mask_new = tf.multiply(mask_inv, mask_flip) # add on mask must be present in the flip and not belong to original
        mask_new = tf.multiply(self.confidence, mask_new) # weight the new mask by the confidence value
        
        xk_flip = tf.image.flip_left_right(tf.image.flip_up_down(xk_sampled_headless))
        
        xk_flip = tf.stack([xk_flip[:,:,:,0], tf.negative(xk_flip[:,:,:,1])], axis=-1) # cplx conjugate
        
        xk_flip_mask = tf.multiply(xk_flip, mask_inv) # only take numbers that were not originally aquired
        
        xk_sampled_update_headless = tf.add(xk_sampled_headless, xk_flip_mask)
        mask_update_headless = tf.add(mask_headless, mask_new)
        
        # concat the head back
        xk_update = tf.concat([xk_sampled[:,1:,0:1,:],xk_sampled_update_headless], axis=2)
        xk_update = tf.concat([xk_sampled[:,0:1,:,:],xk_update], axis=1)
        
        mask_update = tf.concat([mask[:,1:,0:1,:], mask_update_headless], axis=2)
        mask_update = tf.concat([mask[:,0:1,:,:], mask_update], axis=1)
        
        # temporarily convert xk_update to image domain, check oriention
        # xk_update = tf.image.flip_up_down(tf.image.flip_left_right(xk_update))
        # xk_update_im = tf.ifft2d((tf.complex(xk_update[:,:,:,0],xk_update[:,:,:,1])))
        # output=xk_update_im
        
        output = tf.concat([mask_update, xk_update], axis=-1)
        
        return output

    def compute_output_shape(self, input_shapes):

        # shape from (?,256,256,6) to (?,256,256,2)
        output_shape = (input_shapes[0], input_shapes[1], input_shapes[2], 2)
        return output_shape # last dimension comes from stacking img, mask and img_sampled
        
class data_consistency_with_mask_layer(Layer):

    ''' Data consistency layer with mask'''
    # input: concatenated image(from pipeline), mask and xk_sample
    # output: image with k-space sample replaced with xk_sample to ensure data consistency

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

        x = inputs[...,0:2] # type tensor
        mask = inputs[...,2:4]
        xk_sampled = inputs[...,4:6]

        # if self.lam:  # noisy case
        #     output = (x + self.lam * xk_sampled) / (1 + self.lam)
        # else:  # noiseless case, essentially just using the original data
        #     output = (1 - mask) * x + xk_sampled
        output = (1 - mask) * x + xk_sampled * mask

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

    input = Input((256, 256, 4)) # do not consider batch as the first dimension
    output = symmetry_with_mask_layer()(input)

    model = Model(inputs=input, outputs=output)

    data_in = np.zeros((10,256,256,4))
    data_in[:,125,124,:2] = 1.0
    
    data_in[:,125,124,2:4] = 2.0

    data_out = model.predict(data_in)
    pdb.set_trace()
