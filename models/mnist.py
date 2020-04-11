from tensorflow.keras import layers
import tensorflow as tf
import datetime


class MnistModel(tf.keras.Sequential):
    
    def __init__(self):
        super(MnistModel, self).__init__()
        self.flatten = layers.Flatten()
        self.d1 = layers.Dense(128, activation=tf.keras.activations.relu)
        self.dropout = layers.Dropout(0.2)
        self.d2 = layers.Dense(10, activation=tf.keras.activations.softmax)

    def call(self, inputs, training=None, mask=None):
        inputs = self.flatten(inputs)
        inputs = self.d1(inputs)
        inputs = self.dropout(inputs)
        inputs = self.d2(inputs)
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
            filepath='saved_params/mnist/checkpoints/mnist.ckpt',
            save_best_only=False,
            verbose=1,
            save_weights_only=True
        )
    
        tb_callback = tf.keras.callbacks.TensorBoard(
            log_dir='saved_params/mnist/tensorboard/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
            histogram_freq=1
        )
    
        cbs = [cp_callback, tb_callback]
        return cbs
