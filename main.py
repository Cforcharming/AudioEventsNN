from models import cnn, bcnn, inception_v3, vgg
from datasets import mivia_2
import tensorflow as tf
import logging


def _prepare_data(db, snr=None):
    if db == 'mivia':
        return mivia_2.load_data(db_level=snr)
    else:
        raise ValueError('Only mivia is accepted as database.')


def _construct_network(net):
    if net == 'cnn':
        return cnn.CnnModel()
    elif net == 'vgg':
        return vgg.VggModel()
    elif net == 'bcnn':
        return bcnn.BatchedCnnModel()
    elif net == 'v3':
        return inception_v3.InceptionV3Model()
    else:
        raise ValueError('Only cnn, bcnn, v3 or vgg are accepted as model.')


def perform_train():
    info = infos.split(' ')
    logger.info('Preparing dataset %s and constructing model %s with epoch %d...' % (info[0], info[1], int(info[2])))
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        
        train_ds, test_ds = _prepare_data(info[0])
        
        model = _construct_network(info[1])
        model.compile(optimizer=model.optimizer_obj, loss=model.loss_obj, metrics=model.metrics_obj)
        
        try:
            for ix in range(0, int(info[2]) + 1, 5):
                latest = tf.train.latest_checkpoint('saved_params/%s/checkpoints/' % info[1])
                if latest:
                    model.load_weights(latest)
                    logger.info('restored latest')
                logger.info('start training')
                model.fit(x=train_ds,
                          epochs=5,
                          verbose=1,
                          validation_data=test_ds,
                          shuffle=True,
                          callbacks=model.cbs
                          )
        
        except KeyboardInterrupt:
            logger.info('Stopped by KeyboardInterrupt.')
        
        except Exception as ex:
            logger.error(ex)
        
        finally:
            logger.info('Done training.')


def perform_evaluate():
    info = infos.split(' ')
    logger.info('Preparing dataset %s and testing model %s' % (info[0], info[1]))
    
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        
        model = _construct_network(info[1])
        model.compile(optimizer=model.optimizer_obj, loss=model.loss_obj, metrics=model.metrics_obj)
        model.build(input_shape=[32, 128, 128, 1])
        
        # latest = 'saved_params/bcnn/checkpoints/0011_ckpt' 17
        latest = tf.train.latest_checkpoint('saved_params/%s/checkpoints/' % info[1])
        model.load_weights(latest).expect_partial()
        for db_level in range(5, 31, 5):
            train_ds, test_ds = _prepare_data(info[0], db_level)
            
            loss, acc = model.evaluate(x=test_ds, steps=105, verbose=1)
            
            logger.info("Model %s accuracy on dataset %s for SNR=%d: %f" % (info[1], info[0], db_level, acc))
        model.save_weights('saved_params/%s/models/final_ckpt' % info[1])
        logger.info('Done evaluating %s' % info[1])


def v():
    logger.info('Preparing dataset mivia and constructing model v3 with epoch 30')
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        
        train_ds, test_ds = _prepare_data('mivia')
        
        model = _construct_network('v3')
        model.v3.compile(optimizer=model.optimizer_obj, loss=model.loss_obj, metrics=model.metrics_obj)
        
        try:
            for ix in range(0, 31, 5):
                latest = tf.train.latest_checkpoint('saved_params/v3/checkpoints/')
                if latest:
                    model.v3.load_weights(latest)
                    logger.info('restored latest')
                logger.info('start training')
                model.v3.fit(x=train_ds,
                             epochs=5,
                             verbose=1,
                             validation_data=test_ds,
                             shuffle=True,
                             callbacks=model.cbs
                             )

        except KeyboardInterrupt:
            logger.info('Stopped by KeyboardInterrupt.')

        except Exception as ex:
            logger.error(ex)

        finally:
            logger.info('Done training.')

        # try:
        #     model.v3.load_weights('saved_params/v3/checkpoints/0004_ckpt')
        #     model.save_weights('saved_params/v3/models/final_ckpt')
        #
        #     for i in range(5, 31, 5):
        #         logger.info('Evaluating performance on %ddB OF SNR' % i)
        #         train_ds, test_ds = _prepare_data('mivia', i)
        #         loss, acc = model.v3.evaluate(x=train_ds, verbose=1)
        #         logger.info("Model %s accuracy on dataset %s for SNR=%d: %5.2f" % ('v3', 'mivia', i, acc))
        #
        # except KeyboardInterrupt:
        #     logger.info('Stopped by KeyboardInterrupt.')
        #
        # except Exception as ex:
        #     logger.error(ex)
        #
        # finally:
        #     logger.info('Done evaluating.')


if __name__ == '__main__':
    fmt = '%(levelname)s %(asctime)s Line %(lineno)s (LOGGER): %(message)s'
    logging.basicConfig(level=logging.INFO, format=fmt)
    logger = logging.getLogger('AudioEventsNN')
    
    infos = 'mivia vgg 30'
    
    # perform_train()
    # perform_evaluate()
    v()
