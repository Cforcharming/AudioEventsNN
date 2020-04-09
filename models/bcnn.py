from __future__ import absolute_import, division, print_function, unicode_literals
from models.model_gen import ModelGen
from tensorflow.keras import layers


class BatchedCnnModel(ModelGen):
    
    def __init__(self):
        super(BatchedCnnModel, self).__init__()
        self.bn1 = layers.BatchNormalization(name='bn1')
        self.bn2 = layers.BatchNormalization(name='bn2')
        self.c1 = layers.Conv2D(filters=32,
                                kernel_size=9,
                                input_shape=(32, 1, 128, 128),
                                data_format='channels_first',
                                activation='relu',
                                name='c1'
                                )
        self.c2 = layers.Conv2D(filters=32,
                                kernel_size=(9, 9),
                                activation='relu',
                                name='c2'
                                )
        self.do1 = layers.Dropout(0.2, name='do1')
        self.do2 = layers.Dropout(0.2, name='do2')
        self.mp1 = layers.MaxPool2D(2, name='mp1')
        self.mp2 = layers.MaxPool2D(2, name='mp2')
        self.flatten = layers.Flatten(name='flat')
        self.d1 = layers.Dense(4, activation='softmax', name='dense')

    def call(self, inputs, training=None, mask=None):
        inputs = self.bn1(inputs)
        inputs = self.c1(inputs)
        inputs = self.bn2(inputs)
        inputs = self.mp1(inputs)
        inputs = self.do1(inputs)
        inputs = self.c2(inputs)
        inputs = self.mp2(inputs)
        inputs = self.do2(inputs)
        inputs = self.flatten(inputs)
        inputs = self.d1(inputs)
        return inputs
