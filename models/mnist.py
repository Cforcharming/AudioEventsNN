from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras import layers
from models.model_gen import ModelGen
import logging
import time


class Mnist(ModelGen):
    
    def __init__(self):
        super(Mnist, self).__init__()
        self.model = None
        self.loss_object = None
        self.optimizer = None
        self.train_loss = None
        self.train_accuracy = None
        self.test_loss = None
        self.test_accuracy = None
    
    def train(self, prepared_data):
        
        fmt = '(%(levelname)s) %(asctime)s Line %(lineno)s: %(message)s'
        logging.basicConfig(level=logging.INFO, format=fmt)
        logger = logging.getLogger('AudioEventsNN')
        
        (x_train, y_train), (x_test, y_test) = prepared_data
        
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            
            self.model = MNISTModel()
            self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
            self.optimizer = tf.keras.optimizers.Adam()
            self.train_loss = tf.keras.metrics.Mean(name='train_loss')
            self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
            self.test_loss = tf.keras.metrics.Mean(name='test_loss')
            self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
            
            epochs = 5
            for epoch in range(epochs):
                start = time.time()
                self.train_loss.reset_states()
                self.train_accuracy.reset_states()
                self.test_loss.reset_states()
                self.test_accuracy.reset_states()
                
                logger.info('Start training epoch %d' % (epoch + 1))
                for (x_train_frame, y_train_label) in zip(x_train, y_train):
                    self.train_step(x_train_frame, y_train_label)
                
                logger.info('Start testing epoch %d' % (epoch + 1))
                for (x_test_frame, y_test_label) in zip(x_test, y_test):
                    self.test_step(x_test_frame, y_test_label)
                
                end = time.time()
                template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}, Take time: {} seconds.'
                logger.info(template.format(epoch + 1,
                            self.train_loss.result(),
                            self.train_accuracy.result() * 100,
                            self.test_loss.result(),
                            self.test_accuracy.result() * 100),
                            (end - start)
                            )
    
    @tf.function
    def train_step(self, x_train_frame, y_train_label):
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            with tf.GradientTape() as tape:
                predictions = self.model(x_train_frame)
                loss = self.loss_object(y_train_label, predictions)
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            
            self.train_loss(loss)
            self.train_accuracy(y_train_label, predictions)
    
    @tf.function
    def test_step(self, x_test_frame, y_test_label):
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            predictions = self.model(x_test_frame)
            t_loss = self.loss_object(x_test_frame, predictions)
            
            self.test_loss(t_loss)
            self.test_accuracy(y_test_label, predictions)


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
