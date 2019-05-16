from torch import nn
import torch

class MLPNet(nn.Module):
    """
    Creates Multilayer Perceptron.
    
    :param in_features: number of input features
    :param out_features: number of output nodes (or Classes)
    :param n_layers: Number of hidden layers in the network
    :param n_hidden: Number of neurons in a given layer
    """
    def __init__(self, in_features, out_features, n_layers, n_hidden):
        super(MLPNet, self).__init__()
        layers = []
        layers.append(nn.Linear(in_features, n_hidden))
        layers.append(nn.ReLU())
        
        for _ in range(n_layers - 2):
            layers.append(nn.Linear(n_hidden, n_hidden))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(n_hidden, out_features))
        self.model = nn.Sequential(*layers)
        
    def forward(self, x_in):
        """
        Performs forward pass for the given input: shape should be [batch_size, in_features]
        """
        x_out = self.model(x_in)
        return x_out
