from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from models.model_gen import ModelGen


class CNN(ModelGen):
    
    def __init__(self):
        super(CNN, self).__init__()
        # TODO
        pass
    
    def train(self, prepared_data):
        # TODO
        pass
    
    @tf.function
    def train_step(self, x_train_frame, y_train_label):
        # TODO
        pass
    
    @tf.function
    def test_step(self, x_test_frame, y_test_label):
        # TODO
        pass


class CNNModel(tf.keras.Model):
    
    def __init__(self):
        super(CNNModel, self).__init__()
        # TODO
    
    def call(self, inputs, training=None, mask=None):
        # TODO
        return inputs
