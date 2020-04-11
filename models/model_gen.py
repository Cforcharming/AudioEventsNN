import tensorflow as tf


class ModelGen(tf.keras.Sequential):
    """
    Inherited from tf.keras.Sequential.
    Generate the General model, including loss,
    optimizer, metrics, callbacks, with structure and call() empty.
    """
    def __init__(self):
        super(ModelGen, self).__init__()
        pass  # add custom layers here
    
    def call(self, inputs, training=None, mask=None):
        pass  # call custom layers here
        return inputs
    
    @property
    def loss_obj(self):
        return 'sparse_categorical_crossentropy'
    
    @property
    def optimizer_obj(self):
        return tf.keras.optimizers.Adam()
    
    @property
    def metrics_obj(self):
        return ['accuracy']
    
    @property
    def cbs(self):
        raise NotImplementedError('cbs for module ModelGen is not implemented.')
