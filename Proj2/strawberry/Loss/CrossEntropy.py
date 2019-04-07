from ..Module.base_module import BaseModule
from torch.autograd import Variable
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
        
        print("targets:")
        print(targets)
        
        print("input:")
        print(inputs)
        
        print("true log softmax:")
        print(inputs.log_softmax(1))
        
        max_inp = inputs.max(1)[0]    
        #substract max from logits for numerical efficiency
        inputs_norm = (inputs.t() - max_inp.view(1, -1)).t()
        log_softmax = (inputs_norm.t() - torch.log(torch.exp(inputs_norm).sum(dim=1))).t()

        
        print("my log softmax:")
        print(log_softmax)
        
        
        loss = torch.nn.NLLLoss()
        print("true loss")
        
        input_tens = Variable(inputs.log_softmax(1).data, requires_grad=True)
        
        with torch.enable_grad():
            output = loss(input_tens, targets.max(1)[1])
        print(output)
        output.backward()
       
        
        print("my loss")
        ce_loss = -(log_softmax * targets).sum()/N
        print(ce_loss)
        
        print("true_gradient:")
        print(input_tens.grad)
        
        return ce_loss

    def backward(self):
        """
        Computes gradients with respect to the input of this module and returns it
        """
        
        
    def __grad_soft_max(self):
        
        
    
    def __call__(self, prediction, labels):
        """
        function object that recieves predictions and labels
        
        :param prediction: Tensor of shape [batch_size, out_features]
        :param labels: Tensor of shape [batch_size, out_features]
        """
        
        #self.data = prediction - labels
        return self._compute_loss(prediction, labels)