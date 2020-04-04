from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras import layers
from models.model_gen import ModelGen
import logging
import time


class Mnist(ModelGen):
    
    def __init__(self):
        super(Mnist, self).__init__()
    
    def train(self, prepared_data):
        
        fmt = '(%(levelname)s) %(asctime)s Line %(lineno)s: %(message)s'
        logging.basicConfig(level=logging.INFO, format=fmt)
        logger = logging.getLogger('AudioEventsNN')
        
        (x_train, y_train), (x_test, y_test) = prepared_data
        
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            
            model = MNISTModel()
            loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
            optimizer = tf.keras.optimizers.Adam()
            train_loss = tf.keras.metrics.Mean(name='train_loss')
            train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
            test_loss = tf.keras.metrics.Mean(name='test_loss')
            test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

            @tf.function
            def train_step(xtf, ytl):
                with tf.GradientTape() as tape:
                    predictions = model(xtf)
                    loss = loss_object(ytl, predictions)
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
                train_loss(loss)
                train_accuracy(ytl, predictions)

            @tf.function
            def test_step(xtf, ytl):
                predictions = model(xtf)
                t_loss = loss_object(xtf, predictions)
    
                test_loss(t_loss)
                test_accuracy(ytl, predictions)
            
            epochs = 5
            for epoch in range(epochs):
                
                start = time.time()
                train_loss.reset_states()
                train_accuracy.reset_states()
                test_loss.reset_states()
                test_accuracy.reset_states()
                
                logger.info('Start training epoch %d' % (epoch + 1))
                for (x_train_frame, y_train_label) in zip(x_train, y_train):
                    train_step(x_train_frame, y_train_label)
                
                logger.info('Start testing epoch %d' % (epoch + 1))
                for (x_test_frame, y_test_label) in zip(x_test, y_test):
                    test_step(x_test_frame, y_test_label)
                
                end = time.time()
                template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}, Take time: {} seconds.'
                logger.info(template.format(epoch + 1,
                            train_loss.result(),
                            train_accuracy.result() * 100,
                            test_loss.result(),
                            test_accuracy.result() * 100),
                            (end - start)
                            )


class MNISTModel(tf.keras.Model):
    
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.conv1 = layers.Conv2D(32, 3, activation='relu')
        self.flatten = layers.Flatten()
        self.d1 = layers.Dense(128, activation='relu')
        self.d2 = layers.Dense(10, activation='softmax')
    
    def call(self, inputs, training=None, mask=None):
        inputs = self.conv1(inputs)
        inputs = self.flatten(inputs)
        inputs = self.d1(inputs)
        return self.d2(inputs)
