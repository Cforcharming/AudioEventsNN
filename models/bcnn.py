from models.model_gen import ModelGen
from tensorflow.keras import layers
from datetime import datetime
import tensorflow as tf


class BatchedCnnModel(ModelGen):
    
    def __init__(self):
        super(BatchedCnnModel, self).__init__()
        self.conv1 = layers.Conv2D(filters=16,
                                   kernel_size=9,
                                   input_shape=(32, 128, 128, 1),
                                   data_format='channels_last',
                                   name='conv1'
                                   )
        self.conv2 = layers.Conv2D(32, 5, data_format='channels_last', name='conv2')
        self.conv3 = layers.Conv2D(64, 7, data_format='channels_last', name='conv3')
        self.conv4 = layers.Conv2D(128, 5, data_format='channels_last', name='conv4')
        self.relu1 = layers.Activation(activation='relu', name='relu1')
        self.relu2 = layers.Activation(activation='relu', name='relu2')
        self.relu3 = layers.Activation(activation='relu', name='relu3')
        self.relu4 = layers.Activation(activation='relu', name='relu4')
        self.batch1 = layers.BatchNormalization(name='batch1')
        self.batch2 = layers.BatchNormalization(name='batch2')
        self.batch3 = layers.BatchNormalization(name='batch3')
        self.batch4 = layers.BatchNormalization(name='batch4')
        self.dropout1 = layers.Dropout(0.2, name='drop1')
        self.dropout2 = layers.Dropout(0.2, name='drop2')
        self.dropout3 = layers.Dropout(0.2, name='drop3')
        self.max_pooling1 = layers.MaxPool2D(2, name='pooling1')
        self.max_pooling2 = layers.MaxPool2D(2, name='pooling2')
        self.max_pooling3 = layers.MaxPool2D(2, name='pooling3')
        self.flatten = layers.Flatten(name='flat')
        self.dense = layers.Dense(4, activation='softmax', name='dense')

    def call(self, inputs, training=None, mask=None):

        inputs = self.conv1(inputs)
        inputs = self.batch1(inputs)
        inputs = self.relu1(inputs)
        inputs = self.max_pooling1(inputs)
        inputs = self.dropout1(inputs)
        
        inputs = self.conv2(inputs)
        inputs = self.batch2(inputs)
        inputs = self.relu2(inputs)
        inputs = self.max_pooling2(inputs)
        inputs = self.dropout2(inputs)

        inputs = self.conv3(inputs)
        inputs = self.batch3(inputs)
        inputs = self.relu3(inputs)
        inputs = self.max_pooling2(inputs)
        inputs = self.dropout2(inputs)
        
        inputs = self.conv4(inputs)
        inputs = self.batch4(inputs)
        inputs = self.relu4(inputs)
        inputs = self.max_pooling3(inputs)
        inputs = self.dropout3(inputs)

        inputs = self.flatten(inputs)
        inputs = self.dense(inputs)
    
        return inputs

    @property
    def cbs(self):
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath='saved_params/bcnn/checkpoints/{epoch:04d}_ckpt',
            verbose=1,
            save_weights_only=True,
        )
        tb_callback = tf.keras.callbacks.TensorBoard(
            log_dir='saved_params/bcnn/tensorboard/' + datetime.now().strftime("%Y%m%d-%H%M%S"),
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
