from __future__ import absolute_import, division, print_function, unicode_literals
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
        return 'adam'
    
    @property
    def metrics_obj(self):
        return ['accuracy']
    
    @property
    def cbs(self):
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath='saved_params/cnn/checkpoints/{epoch:04d}_ckpt',
            verbose=1,
            save_weights_only=True,
        )
        tb_callback = tf.keras.callbacks.TensorBoard(
            log_dir='saved_params/cnn/tensorboard',
            histogram_freq=20,
            write_graph=True,
            update_freq=1000
        )
        es_callback = tf.keras.callbacks.EarlyStopping(
            monitor=self.loss_obj,
            min_delta=0.05,
            patience=10,
            restore_best_weights=True
        )
        
        cbs = [cp_callback, tb_callback, es_callback]
        return cbs
