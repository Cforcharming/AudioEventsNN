import tensorflow as tf


class ModelGen(tf.keras.Model):
    #
    #     def __init__(self):
    #         pass
    #
    #     def train(self, prepared_data):
    #         pass
    #
    #     @tf.function
    #     def train_step(self, x_train_frame, x_test_label):
    #         pass
    #
    #     @tf.function
    #     def test_step(self, x_test_frame, y_test_label):
    #         pass

    def __init__(self):
        super(ModelGen, self).__init__()
        pass
    
    def call(self, inputs, training=None, mask=None):
        super(ModelGen, self).call(inputs, training, mask)
