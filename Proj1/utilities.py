import torch.optim as optim
import torch.nn as nn
import torch

# Since this is a classification problem we use cross entropy loss
def train_model(model, train_input, train_target, mini_batch_size=50, train_classes=None,
                val_input=None, val_target=None, val_classes=None, aux_param = 0.5, n_epoch=200, verbose=False):
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
            }
    
    n_batches = train_input.shape[0]//mini_batch_size
    
    for epoch in range(n_epoch):
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
            model.eval()
            train_acc += get_accuracy(model, batch_input, batch_target).item()
            model.train()
            sum_loss += loss.item()
            
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
            
        history['train_loss'].append(sum_loss/n_batches)
        history['train_acc'].append(100*train_acc/n_batches)
        history['val_loss'].append(loss_val)
        history['val_acc'].append(val_acc)
        
        if verbose:
            print("Epoch: {}, Training loss: {:.4f}, Training accuracy: {:.2f}%, Validation loss: {:.4f}, Validation Accuracy: {:.2f}%".format(
                    epoch, sum_loss/n_batches, 100*train_acc/n_batches, loss_val, 100*val_acc))
        
    return history
        
def get_predictions(model, test_inputs):
    output = model(test_inputs)
    return output.narrow(1, 0, 2).max(1)[1]


def get_accuracy(model, test_inputs, test_targets):
    predictions = get_predictions(model, test_inputs)
    return torch.mean(torch.eq(predictions,test_targets).float())
    