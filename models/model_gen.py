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

    def __init__(self, loss_obj=None, optimizer_obj=None, metrics_obj=None, cbs=None):
        
        super(ModelGen, self).__init__()
        
        self.loss_obj = loss_obj
        self.optimizer_obj = optimizer_obj
        self.metrics_obj = metrics_obj
        self.cbs = cbs
    
    def call(self, inputs, training=None, mask=None):
        return inputs
    
    @property
    def loss_obj(self):
        return self.loss_obj
    
    @property
    def optimizer_obj(self):
        return self.optimizer_obj
    
    @property
    def metrics_obj(self):
        return self.metrics_obj
    
    @property
    def cbs(self):
        return self.cbs
    
    @loss_obj.setter
    def loss_obj(self, new_loss):
        self.loss_obj = new_loss
    
    @optimizer_obj.setter
    def optimizer_obj(self, new_optimizer):
        self.optimizer_obj = new_optimizer
    
    @metrics_obj.setter
    def metrics_obj(self, new_metrics):
        self.metrics_obj = new_metrics
    
    @cbs.setter
    def cbs(self, new_cbs):
        self.cbs = new_cbs
