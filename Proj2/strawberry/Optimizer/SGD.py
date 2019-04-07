from .base_optimizer import Optimizer

class SGD(Optimizer):
    """
    Vanilla Stochastic Gradient Descent Implementation
    
    :param model: Should be one of Linear or Sequential
    :param loss: Should be of type MSE or CrossEntropy
    :param lr: learning rate of stochastic gradient descent
    """
    
    def __init__(self, model, loss, lr=0.01):
        self.model = model
        self.loss = loss
        self.lr = lr
    
    def step(self, inputs, labels):
        """
        Computes one gradient steps of loss and applies gradients to model params.
        """
        # Forward pass:
        predictions = self.model.forward(inputs)
        
        # Forward predictions to loss function
        loss = self.loss(predictions, labels)
        
        #Backward of loss so that gradients are accumulated in it. Then backword of Model to accumulate its gradients.
        self.model.backward(self.loss.backward())
        
        for param in self.model.param():
            param[0] -= self.lr*param[1]
            
        return loss