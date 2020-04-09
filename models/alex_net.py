from __future__ import absolute_import, division, print_function, unicode_literals
from models.model_gen import ModelGen
from tensorflow.keras import layers


class AlexNetModel(ModelGen):
    
    def __init__(self):
        super(AlexNetModel, self).__init__()
    
    def call(self, inputs, training=None, mask=None):
        super().call(inputs, training, mask)
