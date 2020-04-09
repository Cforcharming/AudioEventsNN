from models.cnn import CnnModel


class BatchedCnnModel(CnnModel):
    
    def __init__(self):
        super(BatchedCnnModel, self).__init__()
    
    def call(self, inputs, training=None, mask=None):
        super().call(inputs, training, mask)
