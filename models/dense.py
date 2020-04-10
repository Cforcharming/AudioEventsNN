from __future__ import absolute_import, division, print_function, unicode_literals
from models.model_gen import ModelGen
from tensorflow.keras import layers


class DenseModel(ModelGen):
    
    def __init__(self):
        super(DenseModel, self).__init__()
    
    def call(self, inputs, training=None, mask=None):
        pass
