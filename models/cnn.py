from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow.keras import layers
import tensorflow as tf
import datetime


class CnnModel(tf.keras.Model):
    
    def __init__(self):
        super(CnnModel, self).__init__()
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()
        self.c1 = layers.Conv2D(filters=20, kernel_size=9, input_shape=(128, 128), activation='relu')
        self.dropout = layers.Dropout(0.2)
        self.max_pool = layers.MaxPool2D(2)
        self.flatten = layers.Flatten()
        self.d1 = layers.Dense(4, activation='softmax')
        
    def call(self, inputs, training=None, mask=None):
        inputs = self.bn1(inputs)
        inputs = self.c1(inputs)
        inputs = self.bn2(inputs)
        inputs = self.max_pool(inputs)
        inputs = self.dropout(inputs)
        inputs = self.flatten(inputs)
        inputs = self.d1(inputs)
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
            filepath='saved_params/cnn/checkpoints/cp-{epoch:04d}.ckpt',
            verbose=1,
            save_weights_only=True,
        )
        tb_callback = tf.keras.callbacks.TensorBoard(
            log_dir='saved_params/cnn/tensorboard/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
            histogram_freq=20,
            write_graph=True,
            update_freq=1000
        )
        
        cbs = [cp_callback, tb_callback]
        return cbs
