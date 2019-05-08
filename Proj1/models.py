from torch import nn
from torch.nn import functional as F
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


class BlockConvNet(nn.Module):
    """
    Creates Convolutional neural network as mentioned here : \site the link\
    """
    
    def __init__(self, n_filters=8, in_channels=1, kernel_size=3, im_shape=[14, 14], fc_nodes=128, n_classes=10):
        super(BlockConvNet, self).__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 2*n_filters, kernel_size),
            nn.LeakyReLU(),
            nn.Conv2d(2*n_filters, n_filters, kernel_size),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.5)
        )
        
        # fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(n_filters//4*(im_shape[0] - 2*(kernel_size -1))**2, fc_nodes),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(fc_nodes, n_classes)
        )
    
    def forward(self, x_in):
        out = self.model(x_in)
        
        #Flatten output
        out = out.view(x_in.shape[0], -1)
        out = self.fc_layers(out)
        return out

class DeepSiameseNet(nn.Module):
    """
    Creates Deep Siamese network.
    
    :param model: model to share weights
    :param in_features: number of features in the output of model
    """
    
    def __init__(self, model, in_features=20, n_hidden=10):
        super(DeepSiameseNet, self).__init__()        
        self.model = model
        
        self.mlp_layers = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(in_features, n_hidden),
            nn.LeakyReLU(),           
            nn.Linear(n_hidden, 2)
        )        

    def forward(self, x_in):
        """
        x_in shape should be [batch_size, n_channels, width, height]
        """
        
        x_in_0 = x_in.narrow(1, 0, 1)
        x_in_1 = x_in.narrow(1, 1, 1)
        
        # outputs for the two images with weight sharing
        out_0 = self.model(x_in_0)
        out_1 = self.model(x_in_1)
        
        
        #concatenate outputs
        out_concat = torch.cat((out_0, out_1), 1)
        
        #output of MLP layers to classify if img_0 is greater than img_1
        out_mlp = self.mlp_layers(out_concat)
        
        return out_mlp