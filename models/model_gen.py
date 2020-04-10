from __future__ import absolute_import, division, print_function, unicode_literals
from datetime import datetime
import tensorflow as tf


class ModelGen(tf.keras.Sequential):
    """
    Inherited from tf.keras.Sequential.
    Generate the General model, including loss,
    optimizer, metrics, callbacks, with structure and call() empty.
    """
    def __init__(self):
        super(ModelGen, self).__init__()
        self.n = None
        pass  # add custom layers here
    
    def call(self, inputs, training=None, mask=None):
        pass  # call custom layers here
        return inputs
    
    @property
    def loss_obj(self):
        return 'sparse_categorical_crossentropy'
    
    @property
    def optimizer_obj(self):
        return 'adam'
    
    @property
    def metrics_obj(self):
        return ['accuracy']
    
    @property
    def cbs(self):
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath='saved_params/%s/checkpoints/{epoch:04d}_ckpt' % self.n,
            verbose=1,
            save_weights_only=True,
        )
        tb_callback = tf.keras.callbacks.TensorBoard(
            log_dir='saved_params/%s/tensorboard/' % self.n + datetime.now().strftime("%Y%m%d-%H%M%S"),
            histogram_freq=20,
            write_graph=True,
            update_freq='batch'
        )
        es_callback = tf.keras.callbacks.EarlyStopping(
            monitor=self.metrics_obj,
            min_delta=0.05,
            patience=10,
            restore_best_weights=True
        )
        
        cbs = [cp_callback, tb_callback, es_callback]
        return cbs
