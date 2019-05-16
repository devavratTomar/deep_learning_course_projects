import torch
from .base_module import BaseModule
import torch.distributions.binomial as bn

class Dropout(BaseModule):
    """
    Implements Dropout layer
    
    param p: keep probability of a neuron
    """
    def __init__(self, p=0.5): 
        self.data = None
        if p < 0 or p > 1:
            raise ValueError("Probability has to be between 0 and 1, " "but got {}".format(p))
        self.p = p

    def forward(self, inputs):
        """
        Apply Dropouts on inputs.
        """
        if Dropout.train_flag:
            binomial = bn.Binomial(probs=self.p)
            self.data = binomial.sample(inputs.size())
            return inputs * self.data * (1.0/(self.p))
        return inputs

    def backward(self, gradwrtoutput):
        """
        Performs backward pass through this module.
        
        :param gradwrtoutput: gradient of loss with respect to the output of this module. shape should be [batch_size, out_features]
        
        :return: gradient of loss with respect to the input of this module.
        """
        if Dropout.train_flag:
            gradin = torch.empty(self.data.shape).zero_()
            gradin[self.data > 0] = 1.0
            return gradin*gradwrtoutput
        return gradwrtoutput

    def param(self):
        return []