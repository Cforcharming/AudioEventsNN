import tensorflow as tf


class ModelGen(tf.keras.Model):

    def __init__(self):
        super(ModelGen, self).__init__()
        pass
    
    def call(self, inputs, training=None, mask=None):
        super(ModelGen, self).call(inputs, training, mask)
