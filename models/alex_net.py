from models.cnn import CnnModel


class AlexNetModel(CnnModel):
    
    def __init__(self):
        super(AlexNetModel, self).__init__()
    
    def call(self, inputs, training=None, mask=None):
        super().call(inputs, training, mask)
