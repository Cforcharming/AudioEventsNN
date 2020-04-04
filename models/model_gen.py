import tensorflow as tf


class ModelGen:
    
    def __init__(self):
        pass
    
    def train(self, prepared_data):
        pass

    @tf.function
    def train_step(self, x_train_frame, x_test_label):
        pass
    
    @tf.function
    def test_step(self, x_test_frame, y_test_label):
        pass
