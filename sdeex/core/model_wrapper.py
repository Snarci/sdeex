import torch
import torch.nn as nn


    

class ModelWrapper(nn.Module):
    '''
    A wrapper class for models that may require additional functionality
    For example using models form timm or other libraries that can output the list 
    of tensors instead of a output tensor.

    '''
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        #change to match the goal of the model
        return self.model(x)

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))



#main 
if __name__ == '__main__':
    #assert
    import os
    model = ModelWrapper(nn.Linear(10, 10))
    print(model(torch.randn(10)).shape)
    model.save('model.pth')
    model.load('model.pth')
    os.remove('model.pth')
    