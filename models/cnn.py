from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow.keras import layers
import tensorflow as tf
import datetime


class CnnModel(tf.keras.Model):
    
    def __init__(self):
        super(CnnModel, self).__init__()
        # self.c = layers.Conv2D((19, 513))
        
    def call(self, inputs, training=None, mask=None):
        
        return inputs
    
    @property
    def loss_obj(self):
        return 'sparse_categorical_crossentropy'
    
    @property
    def optimizer_obj(self):
        return 'adam'
    
    @property
    def metrics_obj(self):
        return ['accuracy']
    
    @property
    def cbs(self):
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath='saved_params/cnn/checkpoints',
            save_best_only=False,
            verbose=1,
            save_weights_only=True
        )
        tb_callback = tf.keras.callbacks.TensorBoard(
            log_dir='saved_params/cnn/tensorboard' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
            histogram_freq=1
        )
        
        cbs = [cp_callback, tb_callback]
        return cbs
