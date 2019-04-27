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
    cmap = ["red", "green"]
    c = [cmap[int((l.item()+1)//2)] for l in labels]
    plt.scatter(data[:,0], data[:,1], c=c)
    plt.title(title)
    plt.axis('equal')
    plt.show()