from .base_module import BaseModule
import torch

class Sequential(BaseModule):
    """
    Implements MLP with layers Linear, RelU, Tanh.
    """
    def __init__(self, *layers):
        #TODO: check if inputs are of type BaseModule
        self.layers = [layer for layer in layers]
    
    def forward(self, inputs):
        """
        Performs foward pass through all layers.
        """
        out = inputs
        for layer in self.layers:
            out = layer.forward(out)

        return out

    def backward(self, gradwrtoutput):
        """
        Performs backward pass through this module.
        
        :param gradwrtoutput: gradient of loss with respect to the output of this module. shape should be [batch_size, out_features]
        
        :return: gradient of loss with respect to the input of this module.
        """
        grad_out = gradwrtoutput
        for layer in reversed(self.layers):
            grad_out = layer.backward(grad_out)

        return grad_out

    def param(self):
        params = []
        for layer in self.layers:
            for param in layer.param():
                params.append(param)
        return params
    
    def train(self, x_train, y_train, epochs, batch_size, opt, loss, metrics=None, 
              validation_set=None, verbose=0):
        """
        Trains the model given the training data
        
        :param gradwrtoutput: 
        
        :return: gradient of loss with respect to the input of this module.
        """
        n_batches = x_train.shape[0]//batch_size
        
        for epoch in range(epochs):
            
            for batch in range(n_batches):
                
                train_data   = x_train[batch*batch_size:(batch+1)*batch_size, :]
                train_labels = y_train[batch*batch_size:(batch+1)*batch_size, :]
                
                # Forward pass:
                predictions = self.forward(train_data)
                
                # Forward predictions to loss function
                training_loss = loss(predictions, train_labels)
                
                # Backward of loss so that gradients are accumulated in it. Then backword of Model to accumulate its gradients.
                self.backward(loss.backward())
        
                opt.step(self.param())
        
            if verbose:
                print("Epoch: {} Training loss: {}".format(epoch, training_loss))
            
        
    def predict(self, x_test, y_test):  
        """
        Predicst labels and returns one hot vector
        """
        
        model_output = self.forward(x_test)
        
        # Create one hot encoding of predicted labels
        max_idx = torch.argmax(model_output, 1, keepdim=True)
        one_hot = torch.FloatTensor(model_output.shape)
        one_hot.zero_()
        one_hot.scatter_(1, max_idx, 1)
        
        return one_hot       
        
        