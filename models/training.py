from __future__ import absolute_import, division, print_function, unicode_literals
from dataset import mivia_db
from dataset import mnist
import tensorflow as tf
from models.mnist import Mnist
from models.cnn import CNN
import logging


def train(info: str):
    
    fmt = '(%(levelname)s) %(asctime)s Line %(lineno)s: %(message)s'
    logging.basicConfig(level=logging.INFO, format=fmt)
    logger = logging.getLogger('AudioEventsNN')
    
    infos = info.split(' ')
    logger.info('Preparing dataset %s and constructing model %s...' % (infos[0], infos[1]))
    
    (x_train, y_train), (x_test, y_test) = _prepare_data(infos[0])
    
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        
        model = _construct_network(infos[1])
        model.compile(
            optimizer=model.optimizer_obj,
            loss=model.loss_obj,
            metrics=model.metrics_obj
        )
        model.summary()
        model.fit(
            x=x_train,
            y=y_train,
            epochs=5,
            verbose=1,
            shuffle=True,
            callbacks=model.cbs
        )
        model.evaluate(
            x=x_test,
            y=y_test,
            verbose=1
        )
        model.save('../saved_params/mnist/models/mnist.h5')


def _prepare_data(db):
    
    if db == 'mivia':
        return mivia_db.load_data()
    elif db == 'mnist':
        return mnist.load_data()
    else:
        return mivia_db.load_data()


@tf.function
def _construct_network(net):
    
    if net == 'cnn':
        return CNN()
    elif net == 'mnist':
        return Mnist()
    else:
        return CNN()
