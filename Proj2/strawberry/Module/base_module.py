class BaseModule(object):
    """
    This is the base class for all submodules.
    """    
    def forward(self):
        raise NotImplementedError
    
    def backward(self):
        """
        Performs backward pass step. It should get input as gradient of loss with respect to the output of this module if it is not
        the final module.
        This should accumulate the gradients with respect to the parameters (if any) and return a tensor or a tuple of tensors
        containing the gradient of the loss wrt the module’s input.
        """
        raise NotImplementedError
    
    def param(self):
        raise NotImplementedError