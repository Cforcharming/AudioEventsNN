import tensorflow as tf


def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]
    y_train = y_train / 10.
    y_test = y_test / 10.
    return (x_train, y_train), (x_test, y_test)
