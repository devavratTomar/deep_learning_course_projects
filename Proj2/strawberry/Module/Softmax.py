import torch.empty
from .base_module import BaseModule

class Softmax(BaseModule):
    """
    Implement Softmax layer
    
    :param nb_nodes: input and output dimension (number of classes)
    """
    
    def __init__(self, nb_nodes):
        self.nb_nodes = nb_nodes
        
        
    def forward(self, inputs):
        """
        Apply softmax on inputs.
        
        :param inputs: input features tensor of shape [batch_size, nb_nodes]
        
        :return: output of shape [batch_size, nb_nodes]
        """
        
        assert type(inputs) == torch.Tensor
        self.data = inputs
        
        #???? am I allowed to use the method
        return inputs.softmax(self.nb_nodes)
        
        
    def backward(self, gradwrtoutput):
        """
        Performs backward pass step. It should get input as gradient of loss with respect to the output of this module.
        This function accumulate the gradients with respect to the parameters  and return a tensor or a tuple of tensors
        containing the gradient of the loss wrt the module’s input.
        
        :param gradwrtoutput: gradient of loss with respect to the output of this module. shape should be [batch_size, out_features]
        """
        pass
    

    def param(self):
        return []