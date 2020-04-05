import tensorflow as tf
import numpy as np


def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]
    y_train = y_train / 10.
    y_test = y_test / 10.
    # def train_gen():
    #     for (x, y) in zip(x_train, y_train):
    #         yield x, y
    #
    # def test_gen():
    #     for (x, y) in zip(x_test, y_test):
    #         yield x, y
        
    # x_train = tf.data.Dataset.from_tensor_slices(x_train).batch(32)
    # y_train = tf.data.Dataset.from_tensor_slices(y_train).batch(32)
    # x_test = tf.data.Dataset.from_tensor_slices(x_test/255.).batch(32)
    # y_test = tf.data.Dataset.from_tensor_slices(y_test).batch(32)
    
    # train = tf.data.Dataset.from_generator(train_gen, output_types=tf.float32).batch(32)
    # test = tf.data.Dataset.from_generator(test_gen, output_types=tf.float32).batch(32)
    return (x_train, y_train), (x_test, y_test)
