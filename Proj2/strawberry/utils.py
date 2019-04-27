"""
Utils for Strawberry framework
"""
import torch


def get_accuracy(predicted_label, true_label):
    """
    Computes accuracy given predicted and tru labels
    
    :param predicted_label:   predicted labels, one-hot encoding
    :param true_label:        true labels, one-hot encoding
    
    :return:                  accuracy
    """
    
    correct_labels = torch.eq(torch.argmax(predicted_label, 1), torch.argmax(true_label, 1))

    
    return correct_labels.sum().item()  * 1.0 / correct_labels.shape[0]
