import torch.optim as optim
import torch.nn as nn
import torch

# Since this is a classification problem we use cross entropy loss
def train_model(model, train_input, train_target, mini_batch_size=50, train_classes=None, aux_param = 0.5, n_epoch = 25):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(n_epoch):
        sum_loss = 0.0
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
                loss = (nn.CrossEntropyLoss()(output.narrow(1, 0, 2), batch_target) +\
                       aux_param*nn.CrossEntropyLoss()(output.narrow(1, 2, 10), batch_classes.narrow(1, 0, 1).view(-1)) +\
                       aux_param*nn.CrossEntropyLoss()(output.narrow(1, 12, 10), batch_classes.narrow(1, 1, 1).view(-1)))/(2*aux_param + 1.)
                       
            else:
                loss = nn.CrossEntropyLoss()(output, batch_target)
            
            # Update model parameter gradients in the graph
            loss.backward()
            # apply gradients to model parameters
            optimizer.step()
            
            sum_loss += loss.item()
        
        print("Epoch: {}, Training loss: {}".format(epoch, sum_loss/mini_batch_size))
        
        
def get_predictions(model, test_inputs):
    output = model(test_inputs)
    return output.narrow(1, 0, 2).max(1)[1]


def get_accuracy(model, test_inputs, test_targets):
    predictions = get_predictions(model, test_inputs)
    nb_correct = torch.sum(predictions == test_targets, dtype=torch.float32)
    return nb_correct/test_inputs.shape[0]
    