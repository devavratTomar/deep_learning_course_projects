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
    
    
#########################################################################
#            HELPERS/METHODS FOR COMPARISON PLOTS                       #
#########################################################################
    
    
def prepare_standardplot(title, loss, xlabel):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(title, fontsize=18)
    ax1.set_ylabel(loss, fontsize=12)
    ax1.set_xlabel(xlabel, fontsize=12)
    #[ax1.set_ylim(0, 0.3) #was used sometimes for better plotting
    #ax1.set_yscale('log')
    ax2.set_ylabel('accuracy [% correct]',  fontsize=12)
    ax2.set_xlabel(xlabel,  fontsize=12)
    #ax2.set_ylim(86, 100) #was used sometimes for better plotting
    return fig, ax1, ax2


def finalize_standardplot(fig, ax1, ax2):
    ax1handles, ax1labels = ax1.get_legend_handles_labels()
    if len(ax1labels) > 0:
        ax1.legend(ax1handles, ax1labels, prop={'size': 9})
    ax2handles, ax2labels = ax2.get_legend_handles_labels()
    if len(ax2labels) > 0:
        ax2.legend(ax2handles, ax2labels, prop={'size': 9})
    fig.tight_layout()
    plt.subplots_adjust(top=0.9, wspace=0.2)
    

def plot_history(history, title, loss):
    
    fig, ax1, ax2 = prepare_standardplot(title, loss, 'epoch')
    ax1.plot(history['loss'], label = "trainingsdads")
    ax1.plot(history['val_loss'], label = "validation")
    ax2.plot(history['acc'], label = "training", )
    ax2.plot(history['val_acc'], label = "validation")
    finalize_standardplot(fig, ax1, ax2)
    fig.set_figheight(5)
    fig.set_figwidth(15)
    return fig


def comparison_plot(history1, history2, label1, label2, title, fig_size, max_epoch=None):
    fig, ax1, ax2 = prepare_standardplot(title, "loss", "epochs")
    if max_epoch is None:
        max_epoch = len(history1['loss'])
    
    fig.set_figheight(fig_size[0])
    fig.set_figwidth(fig_size[1])
    ax1.plot(history1['loss'][:max_epoch], label=label1 + ' training', alpha=0.6)
    ax1.plot(history1['val_loss'][:max_epoch], label=label1 + ' validation', 
             alpha=0.7, color='violet')
    ax1.plot(history2['loss'][:max_epoch], label=label2 + ' training', alpha=0.5)
    ax1.plot(history2['val_loss'][:max_epoch], label=label2 + ' validation', lw=0.7)
    
    ax2.plot(history1['acc'][:max_epoch], label=label1 + ' training', alpha=0.6)
    ax2.plot(history1['val_acc'][:max_epoch], label=label1 + ' validation',
              alpha=0.7, color='violet')
    ax2.plot(history2['acc'][:max_epoch], label=label2 + ' training', alpha=0.5)
    ax2.plot(history2['val_acc'][:max_epoch], label=label2 + ' validation', lw=0.7)
    #plt.gca().yaxis.set_minor_formatter(ticker.FormatStrFormatter("%.2f"))
    finalize_standardplot(fig, ax1, ax2)
    return fig