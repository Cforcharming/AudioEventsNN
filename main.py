from __future__ import absolute_import, division, print_function, unicode_literals
from datasets import mivia_db, mnist_db
from models import cnn, mnist
import tensorflow as tf
import logging
import time


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


def perform_train():
    
    fmt = '%(levelname)s %(asctime)s Line %(lineno)s (LOGGER): %(message)s'
    logging.basicConfig(level=logging.INFO, format=fmt)
    logger = logging.getLogger('AudioEventsNN')

    info = input('Use database, model and training epochs (split by ' '):').split(' ')
    logger.info('Preparing dataset %s and constructing model %s...' % (info[0], info[1]))
    
    train, test = _prepare_data(info[0])
    
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
        tf.keras.backend.set_learning_phase(1)
        model.fit(
            x=train,
            epochs=int(info[2]),
            verbose=1,
            validation_split=0.2,
            shuffle=True,
            callbacks=model.cbs
        )
        tf.keras.backend.set_learning_phase(0)
        loss, acc = model.evaluate(
            x=test,
            verbose=1
        )
        logger.info("Model accuracy: {:5.2f}%".format(100 * acc))
        model.save('saved_params/%s/cnn-%d.hdf5' % (info[1], int(time.time())))
    

if __name__ == '__main__':
    perform_train()
