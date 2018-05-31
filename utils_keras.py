import tensorflow as tf
import keras
import pdb
from PIL import Image
import io
import numpy as np
''' keras call back function to view predicted image '''

class TensorBoardImage(keras.callbacks.Callback):
    def __init__(self, log_dir, tag, test_ims, num_ims):
        super().__init__() 
        self.log_dir = log_dir
        self.tag = tag
        self.test_ims = test_ims
        self.num_ims = num_ims
        self.writer = tf.summary.FileWriter(self.log_dir)

    def on_epoch_end(self, epoch, logs={}):
        # Load image
        # pdb.set_trace()
        
        if(epoch % 5==0):
            # writing result image every 5 epochs
            
            model_out = self.model.predict(self.test_ims)
            # num_im = model_out.shape[0]

            for k in range(self.num_ims):

                model_out_slice = abs(model_out[k,:,:,0]+model_out[k,:,:,1]*1j)

                # convert model_out to string

                image = Image.fromarray(255*model_out_slice)
                image = image.convert('RGB')
                output = io.BytesIO()
                image.save(output, format='PNG')
                image_string = output.getvalue()
                output.close()
                
                # write summary of image to event file of tensorboard
                
                summary_string = tf.Summary.Image(encoded_image_string=image_string)

                summary = tf.Summary(value=[tf.Summary.Value(tag=self.tag+'_'+str(k), image=summary_string)])

                self.writer.add_summary(summary, epoch)
                
        return
