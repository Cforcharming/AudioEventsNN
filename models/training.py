from __future__ import absolute_import, division, print_function, unicode_literals
from dataset import mivia_db
from dataset import mnist
import tensorflow as tf
from models.mnist import MnistModel
# from models.cnn import CnnModel
import logging


def train(info: str):
    
    fmt = '(%(levelname)s) %(asctime)s Line %(lineno)s: %(message)s'
    logging.basicConfig(level=logging.INFO, format=fmt)
    logger = logging.getLogger('AudioEventsNN')
    
    infos = info.split(' ')
    logger.info('Preparing dataset %s and constructing model %s...' % (infos[0], infos[1]))
    
    (x_train, y_train), (x_test, y_test) = _prepare_data(infos[0])
    
    tf.config.set_soft_device_placement(True)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        
        model = _construct_network(infos[1])
        model.compile(
            optimizer=model.optimizer_obj,
            loss=model.loss_obj,
            metrics=model.metrics_obj
        )

        model.fit(
            x=x_train,
            y=y_train,
            epochs=5,
            verbose=1,
            shuffle=True,
            callbacks=model.cbs
        )
        model.summary()
        model.evaluate(
            x=x_test,
            y=y_test,
            verbose=1
        )
        model.save('../saved_params/%s/models/mnist.h5' % infos[1])


def _prepare_data(db):
    
    if db == 'mivia':
        return mivia_db.load_data()
    elif db == 'mnist':
        return mnist.load_data()
    else:
        return mivia_db.load_data()


def _construct_network(net):
    
    if net == 'cnn':
        # return CnnModel()
        # TODO
        return None
    elif net == 'mnist':
        return MnistModel()
    else:
        # return CnnModel()
        # TODO
        return None
