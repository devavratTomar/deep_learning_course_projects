import torch
import math
import matplotlib.pyplot as plt


def get_labels(data, center):
    """
    Generates Labels for the data as one hot vector
    """
    
    radius = torch.pow(data - center, 2).sum(dim=1)
    labels = torch.empty(data.shape[0], 1).fill_(1)
    
    labels[radius> 1./(2*math.pi)] = 0
        
    return torch.cat((labels, 1-labels), dim=1)


def plot_points(data, labels, title):
    """
    Plots data with labels
    
    :param data:     data set
    :param labels:   corresponding labels in one hot encoding
    :param title:    title of the plot
    """
    cmap = ["red", "green"]
    
    labels_flatten = torch.argmax(labels, 1)
    c = [cmap[int((l.item()+1)//2)] for l in labels_flatten]
    plt.scatter(data[:,0], data[:,1], c=c)
    plt.title(title)
    plt.axis('equal')
    plt.show()