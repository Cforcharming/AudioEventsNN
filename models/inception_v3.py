from tensorflow.keras.applications.inception_v3 import InceptionV3
from models.model_gen import ModelGen
from tensorflow.keras import layers
from datetime import datetime
import tensorflow as tf


class InceptionV3Model(ModelGen):
    
    def __init__(self):
        super(InceptionV3Model, self).__init__()
        v3b = InceptionV3(weights=None, input_shape=(128, 128, 1), include_top=False)
        x = v3b.output
        x = layers.GlobalAveragePooling2D()(x)
        predictions = layers.Dense(4, activation='softmax')(x)
        self._v3 = tf.keras.Model(inputs=v3b.input, outputs=predictions)
    
    @property
    def v3(self):
        return self._v3

    @property
    def cbs(self):
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath='saved_params/v3/checkpoints/{epoch:04d}_ckpt',
            verbose=1,
            save_weights_only=True,
        )
        tb_callback = tf.keras.callbacks.TensorBoard(
            log_dir='saved_params/v3/tensorboard/' + datetime.now().strftime("%Y%m%d-%H%M%S"),
            histogram_freq=20,
            write_graph=True,
            update_freq='batch'
        )
        es_callback = tf.keras.callbacks.EarlyStopping(
            monitor='accuracy',
            min_delta=0.01,
            patience=15,
            restore_best_weights=True
        )
    
        cbs = [cp_callback, es_callback]
        return cbs
