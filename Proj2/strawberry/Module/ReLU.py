import torch.empty
from .base_module import BaseModule

class ReLU(BaseModule):
    """
    Implements ReLU layer
    """
    def __init__(self):
        # Storting where inputs is greater than zero
        self.data = None

    def forward(self, inputs):
        """
        Apply Relu activation on inputs.
        """
        self.data = torch.max(torch.empty(inputs.shape).zero_(), inputs)
        return self.data

    def backward(self, gradwrtoutput):
        gradin = torch.empty(self.data.shape).zero_()
        gradin[self.data > 0] = 1.0
        return gradin*gradwrtoutput

    def param(self):
        return []