from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras import layers


class MnistModel(tf.keras.Sequential):
    
    def __init__(self):
        super(MnistModel, self).__init__()
        
        self.conv1 = layers.Conv2D(32, 3, activation='relu')
        self.flatten = layers.Flatten()
        self.d1 = layers.Dense(128, activation='relu')
        self.d2 = layers.Dense(10, activation='softmax')
        self.dropout = layers.Dropout(0.2)
        self.d3 = layers.Dense(1, activation='relu')

    def call(self, inputs, training=None, mask=None):
        inputs = self.conv1(inputs)
        inputs = self.flatten(inputs)
        inputs = self.d1(inputs)
        inputs = self.dropout(inputs)
        inputs = self.d2(inputs)
        inputs = self.d3(inputs)
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
            filepath='saved_params/mnist/checkpoints',
            save_best_only=False,
            verbose=1,
            save_weights_only=True
        )
    
        tb_callback = tf.keras.callbacks.TensorBoard(log_dir='saved_params/mnist/tensorboard')
    
        cbs = [cp_callback, tb_callback]
        return cbs
