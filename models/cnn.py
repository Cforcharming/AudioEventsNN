from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow.keras import layers
import tensorflow as tf


class CnnModel(tf.keras.Model):
    
    def __init__(self):
        super(CnnModel, self).__init__()
        # self.c = layers.Conv2D((19, 513))
        
    def call(self, inputs, training=None, mask=None):
        
        return inputs
    
    @property
    def loss_obj(self):
        return tf.keras.losses.SparseCategoricalCrossentropy()
    
    @property
    def optimizer_obj(self):
        return tf.keras.optimizers.Adam()
    
    @property
    def metrics_obj(self):
        return [tf.keras.metrics.SparseCategoricalAccuracy(), tf.keras.metrics.AUC()]
    
    @property
    def cbs(self):
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath='saved_params/cnn/checkpoints',
            save_best_only=True,
            verbose=1,
            save_weights_only=True
        )
        
        tb_callback = tf.keras.callbacks.TensorBoard(log_dir='saved_params/cnn/tensorboard')
        
        cbs = [cp_callback, tb_callback]
        return cbs
