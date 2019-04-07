import torch
import matplotlib.pyplot as plt

from strawberry import Module
from strawberry import Loss
from strawberry import Optimizer

import helpers as h

#autograd globally off
#torch.set_grad_enabled ( False )

# testing linear module
def test_linear():
    W = torch.tensor([[5], [1], [-6], [7], [8], [-2]], dtype=torch.float)
    B = -3
    # create dummy data with noisy labels
    TRAIN_FEATURES = torch.empty(1000, 6).uniform_(-1, 1)
    TRAIN_LABELS = torch.mm(TRAIN_FEATURES, W) + B + torch.empty((1000, 1)).normal_(0, 0.5)
    
    lin_layer = Module.Linear(in_features=6, out_features=1)
    loss_fun = Loss.MSE()
    opt = Optimizer.SGD(lin_layer, loss_fun, lr=0.1)
    
    for iter in range(10000):
        mse = opt.step(TRAIN_FEATURES, TRAIN_LABELS)
        if iter//100 == 0:
            print("iter: {}, loss: {}".format(iter, mse))

    print("True weights: {}, {}".format(W, B))
    print("Learned weights: {}, {}".format(lin_layer.W, lin_layer.b))


    
# TODO: implement softmax layer for one hot classification and cross entropy loss functions

def test_sequential():
    TRAIN_FEATURES = torch.empty(1000, 2).uniform_(0, 1)
    TRAIN_LABELS = h.get_labels(TRAIN_FEATURES, torch.empty(1, 2).fill_(0.5))
    
    TEST_FEATURES = torch.empty(1000, 2).uniform_(0, 1)
    
    plt.figure()
    h.plot_points(TRAIN_FEATURES, TRAIN_LABELS[:,0], "Training points and labels")
    
    batch_size = 10
    n_batches = TRAIN_FEATURES.shape[0]//batch_size
    
    
    model = Module.Sequential(Module.Linear(2, 25),
                              Module.ReLU(),
                              Module.Linear(25, 25),
                              Module.ReLU(),
                              Module.Linear(25, 2),
                              Module.Tanh())
    loss_fun = Loss.MSE()
    opt = Optimizer.SGD(model, loss_fun, lr=0.01)
    
    
    for epoch in range(100):
        for batch in range(n_batches):
            train_data = TRAIN_FEATURES[batch*batch_size:(batch+1)*batch_size, :]
            train_labels = TRAIN_LABELS[batch*batch_size:(batch+1)*batch_size, :]
        
            mse = opt.step(train_data, train_labels)
        
        print("Epoch: {} Training loss: {}".format(epoch, mse))
    
        
    test_predictions = model.forward(TEST_FEATURES)
    test_labels = torch.empty(TEST_FEATURES.shape[0]).fill_(-1.0)
    test_labels[test_predictions[:, 0] > 0] = 1.0
    
    plt.figure()
    h.plot_points(TEST_FEATURES, test_labels, "Test points and predictions")
    
    
def test_softmax():
    TRAIN_FEATURES = torch.empty(1000, 2).uniform_(0, 1)
    print(TRAIN_FEATURES.size())
    TRAIN_LABELS = h.get_labels(TRAIN_FEATURES, torch.empty(1, 2).fill_(0.5))
    TRAIN_LABELS = (1+TRAIN_LABELS)/2
    print(TRAIN_LABELS.size())
    
    plt.figure()
    h.plot_points(TRAIN_FEATURES, TRAIN_LABELS[:,0], "Training points and labels")

    
    TEST_FEATURES = torch.empty(1000, 2).uniform_(0, 1)

    batch_size = 10
    n_batches = TRAIN_FEATURES.shape[0]//batch_size
    
    
    model = Module.Sequential(Module.Linear(2, 25),
                              Module.ReLU(),
                              Module.Linear(25, 25),
                              Module.ReLU(),
                              Module.Linear(25, 2))

    
    loss_ce = Loss.CrossEntropy();
    opt = Optimizer.SGD(model, loss_ce, lr=0.01)
    
    for epoch in range(100):
        for batch in range(n_batches):
            train_data = TRAIN_FEATURES[batch*batch_size:(batch+1)*batch_size, :]
            train_labels = TRAIN_LABELS[batch*batch_size:(batch+1)*batch_size, :]
        
            mse = opt.step(train_data, train_labels)
        
        print("Epoch: {} Training loss: {}".format(epoch, mse))
    
        
    test_predictions = model.forward(TEST_FEATURES)
    test_labels = torch.empty(TEST_FEATURES.shape[0]).fill_(-1.0)
    test_labels[test_predictions[:, 0] > 0] = 1.0
    
    plt.figure()
    h.plot_points(TEST_FEATURES, test_labels, "Test points and predictions")
    
test_softmax()