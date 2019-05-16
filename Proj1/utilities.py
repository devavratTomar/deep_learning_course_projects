import torch.optim as optim
import torch.nn as nn
import torch

def train_model(model, train_input, train_target, mini_batch_size=50, train_classes=None,
                val_input=None, val_target=None, val_classes=None, aux_param = 0.5, n_epochs=200, verbose=False):
    """
    This function trains the given model with following parameters.
    
    :param model: Model to train
    :param train_input: Training data. Shape should be [batch_size, ...]
    :param train_target: Training label. 1 if channel 0 image is greater than channel 1 image
    :param mini_batch_size: size of the minibatch
    :param train_classes: Digit class of both channel images. Digit class should be {0, 1, 2... 9}
    :param val_input: Validation Data
    :param val_target: Validation label. 1 if channel 0 image is greater than channel 1 image 
    :param val_class: Digit class of both channel images. Digit class should be {0, 1, 2... 9}
    :param aux_param: The value of auxiliary loss coefficient
    :param n_epochs: Number of epochs to run the experiment
    :parma verbose: If true, prints training and validaiton loss and accuracy after every epoch.
    """
    
    # For all experiments we use Adam Optimizer with learning rate of 0.001
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Since this is a classification problem we use cross entropy loss
    criterion = nn.CrossEntropyLoss()
    
    # Store training and validation loss and accuracy after every epoch
    history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
            }
    
    n_batches = train_input.shape[0]//mini_batch_size
    
    for epoch in range(n_epochs):
        
        # variables to store average training loss and accuracy
        sum_loss = 0.0
        train_acc = 0.0
        
        for b in range(0, train_input.shape[0], mini_batch_size):
            optimizer.zero_grad()
            
            batch_input = train_input.narrow(0, b, mini_batch_size)
            batch_target = train_target.narrow(0, b, mini_batch_size)
            
            output = model(batch_input)
            # if we want to include auxilary loss
            # we assume first two outputs are train_targets (0, 1)
            # next 10 outputs are train_class for channel 0
            # next 10 outputs are train_class for channel 1
            if type(train_classes) != type(None):
                batch_classes = train_classes.narrow(0, b, mini_batch_size)
                loss = (criterion(output.narrow(1, 0, 2), batch_target) +\
                       aux_param*criterion(output.narrow(1, 2, 10), batch_classes.narrow(1, 0, 1).view(-1)) +\
                       aux_param*criterion(output.narrow(1, 12, 10), batch_classes.narrow(1, 1, 1).view(-1)))/(2*aux_param + 1.)
                       
            else:
                loss = criterion(output, batch_target)
            
            # Update model parameter gradients in the graph
            loss.backward()
            # apply gradients to model parameters
            optimizer.step()
            
            # turn train flag off to re-evalute model on current training batch
            model.eval()
            train_acc += get_accuracy(model, batch_input, batch_target).item()
            model.train()
            sum_loss += loss.item()
        
        # Evaluate model on validation data set
        if type(val_input) != type(None):
            with torch.no_grad():
                model.eval()
                output_val = model(val_input)
                if type(val_classes) != type(None):
                    loss_val = (criterion(output_val.narrow(1, 0, 2), val_target) +\
                    aux_param*criterion(output_val.narrow(1, 2, 10), val_classes.narrow(1, 0, 1).view(-1)) +\
                    aux_param*criterion(output_val.narrow(1, 12, 10), val_classes.narrow(1, 1, 1).view(-1)))/(2*aux_param + 1.)
                else:
                    loss_val = criterion(output_val, val_target).item()
                    
                val_acc = get_accuracy(model, val_input, val_target).item()
                model.train()
        
        # Append the history list with current epoch logss
        history['train_loss'].append(sum_loss/n_batches)
        history['train_acc'].append(100*train_acc/n_batches)
        history['val_loss'].append(loss_val)
        history['val_acc'].append(val_acc)
        
        if verbose:
            print("Epoch: {}, Training loss: {:.4f}, Training accuracy: {:.2f}%, Validation loss: {:.4f}, Validation Accuracy: {:.2f}%".format(
                    epoch, sum_loss/n_batches, 100*train_acc/n_batches, loss_val, 100*val_acc))
        
    return history
        
def get_predictions(model, test_inputs):
    """
    Returns predictions of the given model on test_inputs. Model should be in eval mode
    """
    output = model(test_inputs)
    return output.narrow(1, 0, 2).max(1)[1]


def get_accuracy(model, test_inputs, test_targets):
    """
    Returns accuracy of the given model on test_inputs. test_targets are the true labels. Model should be in eval mode.
    """
    predictions = get_predictions(model, test_inputs)
    return torch.mean(torch.eq(predictions,test_targets).float())
    