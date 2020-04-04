import tensorflow as tf


def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]
    x_train = tf.data.Dataset.from_tensor_slices(x_train).batch(32)
    y_train = tf.data.Dataset.from_tensor_slices(y_train).batch(32)
    x_test = tf.data.Dataset.from_tensor_slices(x_test/255.).batch(32)
    y_test = tf.data.Dataset.from_tensor_slices(y_test).batch(32)
    return (x_train, y_train), (x_test, y_test)
