from datasets import mivia_db
import tensorflow as tf
import numpy as np


def load_data(db_level=None):
    
    def _train_gen():
        if db_level is not None:
            xy = np.load('data/train%02d.npz' % db_level, allow_pickle=True)
            x_train = xy['x']
            y_train = xy['y']
            for (i, j) in zip(x_train, y_train):
                yield i, j
        else:
            for i in range(5, 31, 5):
                xy = np.load('data/train%02d.npz' % i, allow_pickle=True)
                x_train = xy['x']
                y_train = xy['y']
                for (k, j) in zip(x_train, y_train):
                    yield k, j

    # noinspection DuplicatedCode
    def _test_gen():
        if db_level is not None:
            xy = np.load('data/test%02d.npz' % db_level, allow_pickle=True)
            x_test = xy['x']
            y_test = xy['y']
            for (i, j) in zip(x_test, y_test):
                yield i, j
        else:
            for i in range(5, 31, 5):
                xy = np.load('data/test%02d.npz' % i, allow_pickle=True)
                x_test = xy['x']
                y_test = xy['y']
                for (k, j) in zip(x_test, y_test):
                    yield k, j

    train_set = tf.data.Dataset.from_generator(generator=_train_gen,
                                               output_types=(tf.float32, tf.float32),
                                               output_shapes=(tf.TensorShape((128, 128, 1)), tf.TensorShape((1,)))
                                               ).prefetch(tf.data.experimental.AUTOTUNE).batch(32)
    test_set = tf.data.Dataset.from_generator(generator=_test_gen,
                                              output_types=(tf.float32, tf.float32),
                                              output_shapes=(tf.TensorShape((128, 128, 1)), tf.TensorShape((1,)))
                                              ).prefetch(tf.data.experimental.AUTOTUNE).batch(32)

    return train_set, test_set


def trans_data():
    for i in range(5, 31, 5):
        tr, te = mivia_db.load_data(db_level=i)
        tx = []
        ty = []
        for (k, w) in te:
            tx.append(k.numpy())
            ty.append(w.numpy())
        np.savez('data/test%02d.npz' % i, x=tx, y=ty)
