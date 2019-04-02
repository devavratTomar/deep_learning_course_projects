import torch.empty
from .base_module import BaseModule

#TODO: Add GPU support for faster computation

class Linear(BaseModule):
    """
    Implements a linear neural network layer.
    
    :param in_features: number of input features
    :param out_features: number of output features
    :param bias: Bool variable for including bias
    """
    
    def __init__(self, in_features, out_features):
        #TODO: add an initialization module that can initialize parameters according to a distribution
        #TODO: make bias optional?
        std = (2./(in_features*out_features))**0.5
        self.W = torch.empty((in_features, out_features)).normal_(std=std)
        self.b = torch.empty((1, out_features)).zero_()
        
        self.grad_W = torch.empty((in_features, out_features)).zero_()
        self.grad_b = torch.empty((1, out_features)).zero_()
        
        self.data = None
    
    def forward(self, inputs):
        """
        Performs forward pass for the given input.
        
        :param inputs: input features tensor of shape [batch_size, in_features]
        
        :return: output of shape [batch_size, out_features]
        """
        assert type(inputs) == torch.Tensor
        # reset previous gradients?
#        self.grad_W.zero_()
#        self.grad_b.zero_()
#        
        self.data = inputs
        return torch.mm(inputs, self.W) + self.b
        
    def backward(self, gradwrtoutput):
        """
        Performs backward pass step. It should get input as gradient of loss with respect to the output of this module.
        This function accumulate the gradients with respect to the parameters  and return a tensor or a tuple of tensors
        containing the gradient of the loss wrt the module’s input.
        
        :param gradwrtoutput: gradient of loss with respect to the output of this module. shape should be [batch_size, out_features]
        """
        #TODO: prove that the gradients of loss with respect to W and b is the following in report
        
        #gradient accumulation
        self.grad_W = torch.mm(self.data.t(), gradwrtoutput)
        self.grad_b = gradwrtoutput.sum(dim=0, keepdim=True)
        
        # computing gradient of the loss wrt to module's input
        return torch.mm(gradwrtoutput, self.W.t())
        
    def param(self):
        return [[self.W, self.grad_W], [self.b, self.grad_b]]