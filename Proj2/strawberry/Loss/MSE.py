from ..Module.base_module import BaseModule

class MSE(BaseModule):
    """
    Class for computing MSE or L2 Norm
    :param reduce: set this to True to get a average loss per batch
    """
    def __init__(self, reduce=True):
        self.reduce = reduce
    
    def __compute_norm(self, inputs):
        """
        compute norm of inputs
        """
        if self.reduce:
            return inputs.pow(2).sum()/inputs.shape[0]
        else:
            return inputs.pow(2).sum()
    
    def backward(self):
        """
        Computes gradients with respect to the input of this module and returns it
        """
        N = self.data.shape[0] if self.reduce else 1.
        return 2*(self.data)/N 
        
    
    def __call__(self, prediction, labels):
        """
        function object that recieves predictions and labels
        
        :param prediction: Tensor of shape [batch_size, out_features]
        :param labels: Tensor of shape [batch_size, out_features]
        """
        
        self.data = prediction - labels
        return self.__computer_norm(self.data)