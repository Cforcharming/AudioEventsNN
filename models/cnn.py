from __future__ import absolute_import, division, print_function, unicode_literals
from models.model_gen import ModelGen
from tensorflow.keras import layers
from datetime import datetime
import tensorflow as tf


class CnnModel(ModelGen):
    
    def __init__(self):
        super(CnnModel, self).__init__()
        self.n = 'cnn'
        self.conv1 = layers.Conv2D(filters=16,
                                   kernel_size=9,
                                   input_shape=(32, 128, 128, 1),
                                   data_format='channels_last',
                                   activation='relu',
                                   name='conv1'
                                   )
        self.conv2 = layers.Conv2D(filters=32,
                                   kernel_size=5,
                                   data_format='channels_last',
                                   activation='relu',
                                   name='conv2'
                                   )
        self.conv3 = layers.Conv2D(filters=64,
                                   kernel_size=7,
                                   data_format='channels_last',
                                   activation='relu',
                                   name='conv3'
                                   )
        self.conv4 = layers.Conv2D(filters=128,
                                   kernel_size=5,
                                   data_format='channels_last',
                                   activation='relu',
                                   name='conv4'
                                   )
        self.dropout = layers.Dropout(0.2, name='drop')
        self.max_pooling = layers.MaxPool2D(2, name='pooling')
        self.flatten = layers.Flatten(name='flat')
        self.dense = layers.Dense(4, activation='softmax', name='dense')
    
    def call(self, inputs, training=None, mask=None):
        
        inputs = self.conv1(inputs)
        inputs = self.max_pooling(inputs)
        inputs = self.dropout(inputs)
        
        inputs = self.conv2(inputs)
        inputs = self.max_pooling(inputs)
        inputs = self.dropout(inputs)
        
        inputs = self.conv3(inputs)
        inputs = self.max_pooling(inputs)
        inputs = self.dropout(inputs)
        
        inputs = self.conv4(inputs)
        inputs = self.max_pooling(inputs)
        inputs = self.dropout(inputs)
        
        inputs = self.flatten(inputs)
        inputs = self.dense(inputs)
        
        return inputs

    @property
    def cbs(self):
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath='saved_params/cnn/checkpoints/{epoch:04d}_ckpt',
            verbose=1,
            save_weights_only=True,
        )
        tb_callback = tf.keras.callbacks.TensorBoard(
            log_dir='saved_params/cnn/tensorboard/' + datetime.now().strftime("%Y%m%d-%H%M%S"),
            histogram_freq=20,
            write_graph=True,
            update_freq='batch'
        )
        es_callback = tf.keras.callbacks.EarlyStopping(
            monitor=self.metrics_obj,
            min_delta=0.05,
            patience=10,
            restore_best_weights=True
        )
    
        cbs = [cp_callback, tb_callback, es_callback]
        return cbs
