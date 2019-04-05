from ..Module.base_module import BaseModule
import torch

class CrossEntropy(BaseModule):
    """
    Class for computing cross entropy
    First computes softmax and then loss since doing so is numerically more efficient 
    
    :param reduce: set this to True to get a average loss per batch (True by default)
    """
    def __init__(self, reduce=True):
        #ToDo should I include the mean reduction?
        self.reduce = reduce
    
    def forward(self, inputs, targets):
        """
        :param inputs: scores (logits) of each class with shape [batch_size, in_features]
        """
        
        print("targets:")
        print(targets)
        
        print("input:")
        print(inputs)
        
        
        #if reduce == True:
            
        #else:
        
        print("true log softmax:")
        print(inputs.log_softmax(1))
        
        max_inp = inputs.max(1)[0]
        #substract max from logits
        inputs_norm = (inputs.t() - max_inp.view(1, -1)).t()
        log_softmax = (inputs_norm.t() - torch.log(torch.exp(inputs_norm).sum(dim=1))).t()

        
        print("my log softmax:")
        print(log_softmax)
        
        
        #print()
        
        loss = torch.nn.NLLLoss()
        print("true loss")
        print(loss(inputs.log_softmax(1), targets.max(1)[1]))
        
        print("my loss")
        print(-(log_softmax * targets).sum()/10)
        
        return log_softmax
        #return None

    def backward(self):
        """
        Computes gradients with respect to the input of this module and returns it
        """
        
        
    
    def __call__(self, prediction, labels):
        """
        function object that recieves predictions and labels
        
        :param prediction: Tensor of shape [batch_size, out_features]
        :param labels: Tensor of shape [batch_size, out_features]
        """
        
        self.data = prediction - labels
        return self.forward(self.data, labels)