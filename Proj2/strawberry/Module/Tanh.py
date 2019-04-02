import torch.empty
from .base_module import BaseModule

class Tanh(BaseModule):
    def __init__(self):
        # Storting where inputs is greater than zero
        self.data = None

    def forward(self, inputs):
        """
        Apply Tanh activation on inputs.
        """
        self.data = torch.tanh(inputs)
        return self.data

    def backward(self, gradwrtoutput):
        gradin = 1.0 - self.data*self.data
        return gradin*gradwrtoutput

    def param(self):
        return []