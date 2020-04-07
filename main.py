from __future__ import absolute_import, division, print_function, unicode_literals
from datasets import mivia_db, mnist_db
from models import cnn, mnist
import tensorflow as tf
import numpy as np
import logging
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def _prepare_data(db):
    if db == 'mivia':
        return mivia_db.load_data()
    elif db == 'mnist':
        return mnist_db.load_data()
    else:
        return mivia_db.load_data()


def _construct_network(net):
    if net == 'cnn':
        return cnn.CnnModel()
    elif net == 'mnist':
        return mnist.MnistModel()
    else:
        return cnn.CnnModel()


def train():
    
    fmt = '%(levelname)s %(asctime)s Line %(lineno)s (LOGGER): %(message)s'
    logging.basicConfig(level=logging.INFO, format=fmt)
    logger = logging.getLogger('AudioEventsNN')

    info = input('Use database, model and training epochs (split by ' '):').split(' ')
    logger.info('Preparing dataset %s and constructing model %s...' % (info[0], info[1]))
    
    (x_train, y_train), (x_test, y_test) = _prepare_data(info[0])
    
    tf.config.set_soft_device_placement(True)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        
        model = _construct_network(info[1])
        model.compile(
            optimizer=model.optimizer_obj,
            loss=model.loss_obj,
            metrics=model.metrics_obj
        )
        model.fit(
            x=x_train,
            y=y_train,
            epochs=int(info[2]),
            verbose=1,
            shuffle=True,
            callbacks=model.cbs
        )
        score = model.evaluate(
            x=x_test,
            y=y_test,
            verbose=1
        )
        logger.info(score)
        model.save('saved_params/%s/models.h5' % info[1])
    

if __name__ == '__main__':
    main()
