from __future__ import absolute_import, division, print_function, unicode_literals
from models.model_gen import ModelGen
# from tensorflow.keras import layers
from datetime import datetime
import tensorflow as tf


class InceptionV3Model(ModelGen):
    
    def __init__(self):
        super(InceptionV3Model, self).__init__()
    
    def call(self, inputs, training=None, mask=None):
        
        return inputs

    @property
    def cbs(self):
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath='saved_params/v3/checkpoints/{epoch:04d}_ckpt',
            verbose=1,
            save_weights_only=True,
        )
        tb_callback = tf.keras.callbacks.TensorBoard(
            log_dir='saved_params/v3/tensorboard/' + datetime.now().strftime("%Y%m%d-%H%M%S"),
            histogram_freq=20,
            write_graph=True,
            update_freq='batch'
        )
        es_callback = tf.keras.callbacks.EarlyStopping(
            monitor='accuracy',
            min_delta=0.05,
            patience=10,
            restore_best_weights=True
        )
    
        cbs = [cp_callback, tb_callback, es_callback]
        return cbs
