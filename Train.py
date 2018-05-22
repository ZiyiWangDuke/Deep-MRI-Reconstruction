import keras
from keras import models
from keras.layers.core import Activation, Reshape, Permute, Dropout, Dense
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D,Conv2DTranspose
from keras.layers.normalization import BatchNormalization

import pdb
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt

# from imagenet_utils import decode_predictions
from keras.datasets import cifar10

(img_train, label_train), (img_test, label_test) = cifar10.load_data()

# trainning parameters
batch_size=32
epochs=40

''' prepare for input images '''

num_train = 5000
num_test = 400
im_w = 32

y_train = np.expand_dims(img_train[:num_train,:,:,0],axis=3)
y_test = np.expand_dims(img_test[:num_test,:,:,0],axis=3)

x_train = np.zeros((num_train,im_w,im_w,2),dtype='complex64')
x_test = np.zeros((num_test,im_w,im_w,2),dtype='complex64')

for k in range(num_train):
    complex_slice = np.fft.fftshift(np.fft.fft2(y_train[k,:,:,0]))
    x_train[k,:,:,0] = np.real(complex_slice)
    x_train[k,:,:,1] = np.imag(complex_slice)

for k in range(num_test):
    complex_slice = np.fft.fftshift(np.fft.fft2(y_test[k,:,:,0]))
    x_test[k,:,:,0] = np.real(complex_slice)
    x_test[k,:,:,1] = np.imag(complex_slice)

x_train = x_train.reshape((num_train,im_w*im_w*2)).astype('float32')
x_test = x_test.reshape((num_test,im_w*im_w*2)).astype('float32')

''' prepare for callback functions '''

## call back function
class OutputObserver(keras.callbacks.Callback):
    """ callback to observe the output of the after each epoch """
    def __init__(self, test_images):
        self.output = []
        self.test_data = test_images
        images_complex = test_images.reshape((6,im_w,im_w,2))
        self.test_images = images_complex[:,:,:,0] + 1j*images_complex[:,:,:,1]

    def on_epoch_end(self, epoch, logs={}):
        # self.output.append(self.model.predict(self.test_images))
        epoch_out = self.model.predict(self.test_data)[:,:,:,0]
        image_truth = y_test[:6,:,:,0]

        plt_block = np.concatenate((image_truth.reshape(6*im_w,im_w),epoch_out.reshape(6*im_w,im_w)),axis=1)
        plt.imshow(plt_block)
        plt.savefig('Train_Results/epoch_out_at_'+str(epoch))
        # pdb.set_trace()

OutputObserver = OutputObserver(test_images = x_test[:6,:])
tensorboard = keras.callbacks.TensorBoard(log_dir="keras_logs/{}".format(datetime.strftime(datetime.now(), '%Y_%m_%d_%H-%M-%S')),
                                          write_grads=False,
                                          write_images=True,
                                          histogram_freq=0)

''' load model and train '''

from model import recon_encoder
recon_encoder.summary()
recon_encoder.compile(loss="mean_squared_error", optimizer='adam', metrics=['accuracy'],sample_weight_mode='temporal')

history = recon_encoder.fit(  x=x_train,
                      y=y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=1,
                      callbacks=[tensorboard, OutputObserver],
                      validation_split=0.2,
                      shuffle=False,
                      initial_epoch=0)

recon_encoder.save('transLearn_180426.h5')
