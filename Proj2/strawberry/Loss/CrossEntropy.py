from ..Module.base_module import BaseModule
import torch

class CrossEntropy(BaseModule):
    """
    Class for computing cross entropy
    Combines softmax activation with loss computation for numerical efficiency
    
    :param reduce: set this to True to get a average loss per batch (True by default)
    """
    def __init__(self, reduce=True):
        self.reduce = reduce
    
    def _compute_loss(self, inputs, targets):
        """
        :param inputs: scores (logits) of each class with shape [batch_size, in_features]
        :param targets: 
        """
        
        N = inputs.shape[0] if self.reduce else 1.
        
        max_inp = inputs.max(1)[0]    
        #substract max from logits for numerical efficiency
        inputs_norm = (inputs.t() - max_inp.view(1, -1)).t()
        log_softmax = (inputs_norm.t() - torch.log(torch.exp(inputs_norm).sum(dim=1))).t()

        ce_loss = -(log_softmax * targets).sum()/N   
    
        self.softmax = (inputs_norm.exp().t() / inputs_norm.exp().sum(dim=1)).t()
        self.targets = targets
        
        return ce_loss

    def backward(self):
        """
        Computes gradients with respect to the input of this module and returns it
        """
        N = self.softmax.shape[0] if self.reduce else 1.
                
        return (-self.targets + self.softmax)/N
        
        
    
    def __call__(self, prediction, labels):
        """
        function object that recieves predictions and labels
        
        :param prediction: Tensor of shape [batch_size, out_features]
        :param labels: Tensor of shape [batch_size, out_features]
        """
        
        #self.data = prediction - labels
        return self._compute_loss(prediction, labels)