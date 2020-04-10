from __future__ import absolute_import, division, print_function, unicode_literals
from models import cnn, bcnn, alex_net, mnist
from datasets import mivia_db, mnist_db
import tensorflow as tf
import logging


def _prepare_data(db, snr=None):
    if db == 'mivia':
        return mivia_db.load_data(db_level=snr)
    elif db == 'mnist':
        return mnist_db.load_data()
    else:
        raise ValueError('Only mivia or mnist are accepted as database.')


def _construct_network(net):
    if net == 'cnn':
        return cnn.CnnModel()
    elif net == 'mnist':
        return mnist.MnistModel()
    elif net == 'bcnn':
        return bcnn.BatchedCnnModel()
    elif net == 'alex':
        return alex_net.AlexNetModel()
    else:
        raise ValueError('Only cnn, bcnn, alex or mnist are accepted as model.')


def perform_train(ifs):
    
    info = ifs.split(' ')
    logger.info('Preparing dataset %s and constructing model %s with epoch %d...' % (info[0], info[1], int(info[2])))
    
    # tf.config.set_soft_device_placement(True)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        
        train_ds, test_ds = _prepare_data(info[0])
        
        model = _construct_network(info[1])
        model.compile(optimizer=model.optimizer_obj,
                      loss=model.loss_obj,
                      metrics=model.metrics_obj
                      )
        try:
            model.fit(x=train_ds,
                      epochs=int(info[2]),
                      verbose=1,
                      # validation_data=test_ds,
                      # validation_steps=None,
                      shuffle=True,
                      callbacks=model.cbs
                      )
        except KeyboardInterrupt:
            logger.info('Stopped by KeyboardInterrupt.')
        except Exception as e:
            logger.error(e)
        finally:
            logger.info('Saving model as: saved_params/%s/models/%s.h5' % (info[1], info[1]))
            model.save_weights(filepath='saved_params/%s/models/%s.h5' % (info[1], info[1]),
                               save_format='h5'
                               )
            logger.info('Done training.')


def perform_evaluate(ifs, db_level=None):
    
    info = ifs.split(' ')
    logger.info('Preparing dataset %s and testing model %s' % (info[0], info[1]))
    
    tf.config.set_soft_device_placement(True)
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = _construct_network(info[1])
        model.load_weights('saved_params/%s/models/%s.h5' % (info[1], info[1]))
        train_ds, test_ds = _prepare_data(info[0], db_level)
        try:
            # TODO fix ValueError: Unknown Layer: CnnModel
            loss, acc = model.evaluate(
                x=test_ds,
                verbose=1
            )
            if db_level is not None:
                logger.info("Model %s accuracy on dataset %s for SNR=%d: %5.2f" % (info[1], info[0], db_level, acc))
            else:
                logger.info("Model %s accuracy on dataset %s: %5.2f" % (info[1], info[0], acc))
        except ValueError as e:
            logger.error(e)


if __name__ == '__main__':
    
    fmt = '%(levelname)s %(asctime)s Line %(lineno)s (LOGGER): %(message)s'
    logging.basicConfig(level=logging.INFO, format=fmt)
    logger = logging.getLogger('AudioEventsNN')
    
    infos_list = ['mivia cnn 20']
    
    for infos in infos_list:
        perform_train(infos)
        if infos[0] == 'mivia':
            for i in range(5, 31, 5):
                logger.info('Evaluating performance on %ddB OF SNR' % i)
                perform_evaluate(infos, db_level=i)
        else:
            perform_evaluate(infos)
