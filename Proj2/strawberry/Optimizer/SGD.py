from .base_optimizer import Optimizer

class SGD(Optimizer):
    """
    Vanilla Stochastic Gradient Descent Implementation
    
    :param lr: learning rate of stochastic gradient descent
    """
    
    def __init__(self, lr=0.01):
        self.lr = lr
    
    def step(self, parameters):
        """
        Performs SGD step
        
        :param parameters: value and gradient of parameters of the model
        """
         
        for param in parameters:
            param[0] -= self.lr*param[1]
            
