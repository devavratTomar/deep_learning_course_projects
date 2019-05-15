from torch import nn
import torch

class BlockConvNet(nn.Module):
    """
    Creates Convolutional neural network as mentioned here : \site the link\
    """
    
    def __init__(self, n_filters=32, in_channels=1, kernel_size=3, im_shape=[14, 14], fc_nodes=64, n_classes=10):
        super(BlockConvNet, self).__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, n_filters//2, kernel_size),
            nn.LeakyReLU(),
            nn.Dropout2d(0.5),
            nn.Conv2d(n_filters//2, n_filters, kernel_size),
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
    
    def __init__(self, model, in_features=20, n_hidden=20):
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
        
        return torch.cat((out_mlp, out_0, out_1), 1)