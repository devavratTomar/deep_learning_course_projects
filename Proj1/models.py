from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
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
        self.layers = []
        self.layers.append(nn.Linear(in_features, n_hidden))
        
        for _ in range(n_layers - 2):
            self.layers.append(nn.Linear(n_hidden, n_hidden))
        
        self.layers.append(nn.Linear(n_hidden, out_features))
        
    def forward(self, x_in):
        """
        Performs forward pass for the given input: shape should be [batch_size, in_features]
        """
        x_out = x_in
        
        # Computation graph. Apply activation to all layers except last linear layer containing output
        for layer in self.layers[:-1]:
            x_out = F.relu(layer(x_out))
            
        x_out = self.layers[-1](x_out)
        return x_out
        
class ConvNet(nn.Module):
    """
    Creates Convolutional Neural network.
    
    :param n_classes: number of classes for classification
    :param kernel_size: size of convolution kernel
    :param in_channels: number of input channels
    :param n_features: number of feature maps in each layer
    :param n_layers: number of layers
    :param fc_hidden: number of units in fully_connected hidden layer
    """
    
    def __init__(self, n_classes, kernel_size=3, in_channels=2, n_features=32, n_layers=3, fc_hidden=512):
        super(ConvNet, self).__init__()
        padding = (kernel_size-1)//2
        
        self.conv_layers = []
        self.conv_layers.append(nn.Conv2d(in_channels, n_features, kernel_size, padding=padding))
        
        for _ in range(n_layers - 2):
            self.conv_layers.append(nn.Conv2d(n_features, n_features, kernel_size, padding=padding))
        
        self.conv_layers.append(nn.Conv2d(n_features, 16, kernel_size//2))
        
        self.fc1 = nn.Linear(49*(16), fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, n_classes)
        
    def forward(self, x_in):
        x_out = x_in
        for layer in self.conv_layers[:-1]:
            x_out = F.relu(layer(x_out))
        
        #apply pooling
        x_out = F.max_pool2d(x_out, 2, 2)
        x_out = F.relu(self.conv_layers[-1](x_out))
        
        x_out = x_out.view(x_in.shape[0], -1)
        
        x_out = F.relu(self.fc1(x_out))
        x_out = self.fc2(x_out)
        
        return x_out


class SharedMLPModule(nn.Module):
    
    
    def __init__(self, in_features, out_features, n_layers, n_hidden):
        super(SharedMLPModule, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        
        
        self.weights = []
        self.biases  = []
        
        #create first fully connected hidden layer withs shared weights
        self.weights.append(Variable(torch.random_normal([in_features, n_hidden])))
        self.biases.append(Variable(torch.random_normal([n_hidden])))
        
        #ceate n_hidden-2 fully connected layersw with shared weights 
        for _ in range(self.n_layers - 2):
           self.weights.append(Variable(torch.random_normal([n_hidden, n_hidden])))
           self.biases.append(Variable(torch.random_normal([n_hidden])))
           
        #create output layer
        self.weights.append(Variable(torch.random_normal([n_hidden, out_features])))
        self.biases.append(Variable(torch.random_normal([out_features])))
        
    def forward(self, x_in):
        """
        Performs forward pass for the given input: shape should be [batch_size, 2, in_features]
        """
        
        first_image  = x_in[:, 0, :]
        second_image = x_in[:, 1, :]
           
        #input layer
        l1_1 = torch.mm(self.weights[0], first_image) + self.biases[0]
        l1_2 = torch.mm(self.weights[0], second_image) + self.biases[0]
        
        #middle hidden layers
        for i in range(self.n_layers - 2):
           
           l1_1 = torch.mm(self.weights[i+1], l1_1) + self.biases[i+1]
           l1_2 = torch.mm(self.weights[i+1], l1_2) + self.biases[i+1]
           
        #output layer
        l1_1 = torch.mm(self.weights[-1], l1_1) + self.biases[-1]
        l1_2 = torch.mm(self.weights[-1], l1_2) + self.biases[-1]
        
        