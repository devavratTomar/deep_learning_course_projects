import torch
from .base_module import BaseModule

class Tanh(BaseModule):
    def __init__(self):
        self.data = None

    def forward(self, inputs):
        """
        Apply Tanh activation on inputs.
        """
        self.data = torch.tanh(inputs)
        return self.data

    def backward(self, gradwrtoutput):
        """
        Performs backward pass through this module.
        
        :param gradwrtoutput: gradient of loss with respect to the output of this module. shape should be [batch_size, out_features]
        
        :return: gradient of loss with respect to the input of this module.
        """
        gradin = 1.0 - self.data*self.data
        return gradin*gradwrtoutput

    def param(self):
        return []