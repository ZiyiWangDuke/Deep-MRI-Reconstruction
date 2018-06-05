from keras import backend as K
import keras
import pdb
from keras.engine.topology import Layer
import numpy as np
import tensorflow as tf
import scipy.stats as st

def gkern_inv(kernlen=256, nsig=10, ceil_num=1.003):
    
    """Returns a 2D weight array based on Gaussian distribution"""

    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    
    kernel_raw = kernel_raw/np.max(kernel_raw)
    kernel_inv = ceil_num-kernel_raw
    
    return np.float32(kernel_inv)

class kspace_weight_layer(Layer):
    
    ''' weight the k-space before convolution '''
    # input: k-space (center on the edges)
    # output: weighted k-space
    
    def __init__(self, im_w, flag, **kwargs):
        
        # using a simple Gaussian kernel as a weight right now 
        weight = np.fft.fftshift(gkern_inv(kernlen = im_w))
        weight = np.stack([weight,weight], axis=-1)
        
        self.weight = weight
        self.flag = flag
        
        super(kspace_weight_layer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        # Create a trainable weight variable for this layer, confidence of the symetry 
        # self.confidence = self.add_weight(name='confidence',
        #                            # initializer='uniform',
        #                            initializer = keras.initializers.Constant(1.0),
        #                            shape=(),
        #                            trainable=True)
        
        super(kspace_weight_layer, self).build(input_shape)
        
    def call(self, inputs, **kwargs):
        
        if self.flag == 'do_weigh':
            output = tf.multiply(inputs, tf.convert_to_tensor(self.weight))
        else:
            output = tf.div(inputs, tf.convert_to_tensor(self.weight))
        
        return output

    def compute_output_shape(self, input_shapes):
        
        return input_shapes
    
class kspace_padding_layer(Layer):
    
    '''pad the kspace after the locally connected layer'''
    
    # input: k-space (DC son the edge)
    # output: padded k-space
    
    def __init__(self, pad_len=1, **kwargs):
        
        # using a simple Gaussian kernel as a weight right now 
        self.pad_len = pad_len
        
        super(kspace_padding_layer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        
        super(kspace_padding_layer, self).build(input_shape)
        
    def call(self, inputs, **kwargs):
        
        # input_shapes = inputs.shape
        # pad the zeros across the center column and row 
        
#         t_pad = tf.zeros((input_shapes[0],self.img_w-self.pad_len,self.pad_len,input_shapes[3]))
        
#         mid_pt = input_shapes[1]/2
#         out_pad1 = tf.concat([inputs[:,:,:mid_pt,:],t_pad], axis=2)
#         out_pad1 = tf.concat([out_pad1,inputs[:,:,mid_pt:,:]], axis=2)
        
#         t_pad = tf.zeros((input_shapes[0],self.pad_len,self.img_w,input_shapes[3]))
        
#         mid_pt = self.img_w/2
#         out_pad2 = tf.concat([out_pad1[:,:mid_pt,:,:],t_pad], axis=1)
#         out_pad2 = tf.concat([out_pad2,out_pad1[:,mid_pt:,:,:]], axis=1)
        
        output = K.spatial_2d_padding(inputs, padding=((self.pad_len, self.pad_len), (self.pad_len, self.pad_len)))
        return output

    def compute_output_shape(self, input_shapes):
        
        # shape from (?,126,126,2) to (?,128,128,2)
        output_shapes = (input_shapes[0], input_shapes[1]+2*self.pad_len, input_shapes[2]+2*self.pad_len, input_shapes[3])

        return output_shapes
    

if __name__ == "__main__":

    ''' test data_consistency layer '''

    from keras.layers import Input, Conv2D, concatenate
    from keras.models import Model
    import tensorflow as tf
    import pdb

    input = Input((256, 256, 2)) # do not consider batch as the first dimension
    mid = kspace_weight_layer(im_w=256,flag='do_weigh')(input)
    output = kspace_weight_layer(im_w=256,flag='undo_weigh')(mid)

    model = Model(inputs=input, outputs=output)

    data_in = np.ones((10,256,256,2))
    # data_in[:,125,124,:2] = 1.0
    
    data_in[:,125,124,2:4] = 2.0

    data_out = model.predict(data_in)
    pdb.set_trace()