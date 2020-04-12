from models.model_gen import ModelGen
from tensorflow.keras import layers
from datetime import datetime
import tensorflow as tf


class CnnModel(ModelGen):
    
    def __init__(self):
        super(CnnModel, self).__init__()
        self.conv1 = layers.Conv2D(filters=16,
                                   kernel_size=9,
                                   input_shape=(32, 128, 128, 1),
                                   data_format='channels_last',
                                   activation='relu',
                                   name='conv1'
                                   )
        self.conv2 = layers.Conv2D(32, 5, data_format='channels_last', activation='relu', name='conv2')
        self.conv3 = layers.Conv2D(64, 7, data_format='channels_last', activation='relu', name='conv3')
        self.conv4 = layers.Conv2D(128, 5, data_format='channels_last', activation='relu', name='conv4')
        self.dropout1 = layers.Dropout(0.2, name='drop1')
        self.dropout2 = layers.Dropout(0.2, name='drop2')
        self.dropout3 = layers.Dropout(0.2, name='drop3')
        self.dropout4 = layers.Dropout(0.2, name='drop4')
        self.max_pooling1 = layers.MaxPool2D(2, name='pooling1')
        self.max_pooling2 = layers.MaxPool2D(2, name='pooling2')
        self.max_pooling3 = layers.MaxPool2D(2, name='pooling3')
        self.max_pooling4 = layers.MaxPool2D(2, name='pooling4')
        self.flatten = layers.Flatten(name='flat')
        self.dense = layers.Dense(4, activation='softmax', name='dense')
    
    def call(self, inputs, training=None, mask=None):
        
        inputs = self.conv1(inputs)
        inputs = self.max_pooling1(inputs)
        inputs = self.dropout1(inputs)
        
        inputs = self.conv2(inputs)
        inputs = self.max_pooling2(inputs)
        inputs = self.dropout2(inputs)
        
        inputs = self.conv3(inputs)
        inputs = self.max_pooling3(inputs)
        inputs = self.dropout3(inputs)
        
        inputs = self.conv4(inputs)
        inputs = self.max_pooling4(inputs)
        inputs = self.dropout4(inputs)
        
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
            monitor='accuracy',
            min_delta=0.01,
            patience=15,
            restore_best_weights=True
        )
    
        cbs = [cp_callback, tb_callback, es_callback]
        return cbs
