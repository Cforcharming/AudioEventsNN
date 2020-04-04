from __future__ import absolute_import, division, print_function, unicode_literals
from dataset import mivia_db
from dataset import mnist
from models.mnist import Mnist
from models.cnn import CNN
import logging


def train(info: str):
    
    fmt = '(%(levelname)s) %(asctime)s Line %(lineno)s: %(message)s'
    logging.basicConfig(level=logging.INFO, format=fmt)
    logger = logging.getLogger('AudioEventsNN')
    
    infos = info.split(' ')
    logger.info('Preparing dataset %s and constructing model %s...' % (infos[0], infos[1]))
    
    prepared_data = _prepare_data(infos[0])
    model = _construct_network(infos[1])
    model.train(prepared_data)


def _prepare_data(db):
    
    if db == 'mivia':
        return mivia_db.load_data()
    elif db == 'mnist':
        return mnist.load_data()
    else:
        return mivia_db.load_data()


def _construct_network(net):
    
    if net == 'cnn':
        return CNN()
    elif net == 'mnist':
        return Mnist()
    else:
        return CNN()
