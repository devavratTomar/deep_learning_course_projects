from torch import nn
import torch

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
        conv_layers = []
        
        conv_layers.append(nn.Conv2d(in_channels, n_features, kernel_size, padding=padding))
        conv_layers.append(nn.ReLU())
        
        for _ in range(n_layers - 2):
            conv_layers.append(nn.Conv2d(n_features, n_features, kernel_size, padding=padding))
            conv_layers.append(nn.ReLU())
        
        conv_layers.append(nn.Conv2d(n_features, 16, kernel_size//2))
        conv_layers.append(nn.ReLU())
        
        conv_layers.append(nn.MaxPool2d(2,2))
        
        self.conv_model = nn.Sequential(*conv_layers)
        
        self.fc_model = nn.Sequential(
            nn.Linear(49*(16), fc_hidden),
            nn.ReLU(),
            nn.Linear(fc_hidden, n_classes)
        )
    
    def forward(self, x_in):
        x_out = self.conv_model(x_in)
        x_out = x_out.view(x_in.shape[0], -1)
        
        x_out = self.fc_model(x_out)
        return x_out
