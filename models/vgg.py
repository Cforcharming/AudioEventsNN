from models.model_gen import ModelGen
from tensorflow.keras import layers
from datetime import datetime
import tensorflow as tf


class VggModel(ModelGen):
    
    def __init__(self):
        super(VggModel, self).__init__()
        self.conv1 = layers.Conv2D(filters=64,
                                   kernel_size=5,
                                   input_shape=(32, 128, 128, 1),
                                   activation='relu',
                                   data_format='channels_last',
                                   name='conv1')
        self.conv2 = layers.Conv2D(128, 3, data_format='channels_last', activation='relu', name='conv2')
        self.conv3 = layers.Conv2D(256, 3, data_format='channels_last', activation='relu', name='conv4')
        self.conv4 = layers.Conv2D(512, 2, data_format='channels_last', activation='relu', name='conv5')
        self.conv5 = layers.Conv2D(512, 2, data_format='channels_last', activation='relu', name='conv6')
        self.conv6 = layers.Conv2D(512, 2, data_format='channels_last', activation='relu', name='conv7')
        self.conv7 = layers.Conv2D(512, 2, data_format='channels_last', name='conv4')
        self.batch1 = layers.BatchNormalization(name='batch1')
        self.max_pooling1 = layers.MaxPool2D(2, name='pooling1')
        self.max_pooling2 = layers.MaxPool2D(2, name='pooling2')
        self.max_pooling3 = layers.MaxPool2D(2, name='pooling3')
        self.max_pooling4 = layers.MaxPool2D(2, name='pooling4')
        self.max_pooling5 = layers.MaxPool2D(2, name='pooling5')
        self.flatten = layers.Flatten(name='flat')
        self.dense1 = layers.Dense(4096, activation='relu', name='dense1')
        self.dense2 = layers.Dense(512, activation='relu', name='dense2')
        self.dense3 = layers.Dense(4, activation='softmax', name='dense3')
    
    def call(self, inputs, training=None, mask=None):
        
        inputs = self.conv1(inputs)
        inputs = self.batch1(inputs)
        inputs = self.max_pooling1(inputs)
        
        inputs = self.conv2(inputs)
        inputs = self.max_pooling2(inputs)
        
        inputs = self.conv3(inputs)
        inputs = self.max_pooling3(inputs)
        
        inputs = self.conv4(inputs)
        inputs = self.conv5(inputs)
        inputs = self.max_pooling4(inputs)
        
        inputs = self.conv6(inputs)
        inputs = self.conv7(inputs)
        inputs = self.max_pooling5(inputs)
        
        inputs = self.flatten(inputs)
        inputs = self.dense1(inputs)
        inputs = self.dense2(inputs)
        inputs = self.dense3(inputs)
        
        return inputs

    @property
    def cbs(self):
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath='saved_params/vgg/checkpoints/{epoch:04d}_ckpt',
            verbose=1,
            save_weights_only=True,
        )
        tb_callback = tf.keras.callbacks.TensorBoard(
            log_dir='saved_params/vgg/tensorboard/' + datetime.now().strftime("%Y%m%d-%H%M%S"),
            histogram_freq=20,
            write_graph=True,
            update_freq='batch'
        )
        es_callback = tf.keras.callbacks.EarlyStopping(
            monitor='accuracy',
            min_delta=0.01,
            patience=15,
            restore_best_weights=True
        )
    
        cbs = [cp_callback, tb_callback, es_callback]
        return cbs
