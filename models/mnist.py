from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras import layers
# from models.model_gen import ModelGen
# import logging
# import time


# class Mnist(ModelGen):
#
#     def __init__(self):
#         super(Mnist, self).__init__()
#
#     def train(self, prepared_data):
#
#         fmt = '(%(levelname)s) %(asctime)s Line %(lineno)s: %(message)s'
#         logging.basicConfig(level=logging.INFO, format=fmt)
#         logger = logging.getLogger('AudioEventsNN')
#
#         (x_train, y_train), (x_test, y_test) = prepared_data
#
#         strategy = tf.distribute.MirroredStrategy()
#         with strategy.scope():
#
#             model = MNISTModel()
#             # loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
#             # optimizer = tf.keras.optimizers.Adam()
#             # train_loss = tf.keras.metrics.Mean(name='train_loss')
#             # test_loss = tf.keras.metrics.Mean(name='test_loss')
#             model.compile(
#                 optimizer=model.optimizer_obj,
#                 loss=model.loss_obj,
#                 metrics=model.metrics_obj
#             )
#             model.summary()
#             model.fit(
#                 x=x_train,
#                 y=y_train,
#                 epochs=5,
#                 verbose=1,
#                 shuffle=True,
#                 callbacks=model.cbs
#             )
#             model.evaluate(
#                 x=x_test,
#                 y=y_test,
#                 verbose=1
#             )
#             model.save('../saved_params/mnist/models/mnist.h5')
#
#             # @tf.function
#             # def train_step(xtf, ytl):
#             #     with tf.GradientTape() as tape:
#             #         predictions = model(xtf)
#             #         loss = loss_object(ytl, predictions)
#             #     gradients = tape.gradient(loss, model.trainable_variables)
#             #     optimizer.apply_gradients(zip(gradients, model.trainable_variables))
#             #
#             #     train_loss(loss)
#             #     train_accuracy(ytl, predictions)
#             #
#             # @tf.function
#             # def test_step(xtf, ytl):
#             #     predictions = model(xtf)
#             #     t_loss = loss_object(xtf, predictions)
#             #
#             #     test_loss(t_loss)
#             #     test_accuracy(ytl, predictions)
#             #
#             # epochs = 5
#             # for epoch in range(epochs):
#             #
#             #     start = time.time()
#             #     train_loss.reset_states()
#             #     train_accuracy.reset_states()
#             #     test_loss.reset_states()
#             #     test_accuracy.reset_states()
#             #
#             #     logger.info('Start training epoch %d' % (epoch + 1))
#             #     for (x_train_frame, y_train_label) in zip(x_train, y_train):
#             #         train_step(x_train_frame, y_train_label)
#             #
#             #     logger.info('Start testing epoch %d' % (epoch + 1))
#             #     for (x_test_frame, y_test_label) in zip(x_test, y_test):
#             #         test_step(x_test_frame, y_test_label)
#             #
#             #     end = time.time()
#             #  template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}, Take time: {} seconds.'
#             #     logger.info(template.format(epoch + 1,
#             #                 train_loss.result(),
#             #                 train_accuracy.result() * 100,
#             #                 test_loss.result(),
#             #                 test_accuracy.result() * 100),
#             #                 (end - start)
#             #                 )


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
            save_best_only=True,
            verbose=1,
            save_weights_only=True
        )
    
        tb_callback = tf.keras.callbacks.TensorBoard(log_dir='saved_params/mnist/tensorboard')
    
        cbs = [cp_callback, tb_callback]
        return cbs
