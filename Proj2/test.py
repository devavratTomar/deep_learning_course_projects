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
    #labels are either one or minus one
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
    TEST_LABELS   = h.get_labels(TEST_FEATURES, torch.empty(1, 2).fill_(0.5)) 

    
    plt.figure()
    h.plot_points(TRAIN_FEATURES, TRAIN_LABELS[:,0], "Training Data with Labels")
    
    model = Module.Sequential(Module.Linear(2, 25),
                              Module.ReLU(),
                              Module.Linear(25, 25),
                              Module.ReLU(),
                              Module.Linear(25, 2),
                              Module.Tanh())
    loss_fun = Loss.MSE()
    opt = Optimizer.SGD(lr=0.01)
    
    model.train(TRAIN_FEATURES, TRAIN_LABELS, epochs=100, batch_size=10, opt=opt, loss=loss_fun)

        
    test_labels = model.predict(TEST_FEATURES, TEST_LABELS)
    
    plt.figure()
    h.plot_points(TEST_FEATURES, test_labels, "Test points and predictions")
    
    
def test_softmax():
    TRAIN_FEATURES = torch.empty(1000, 2).uniform_(0, 1)
    #print(TRAIN_FEATURES.size())
    TRAIN_LABELS = h.get_labels(TRAIN_FEATURES, torch.empty(1, 2).fill_(0.5))
    #print(TRAIN_LABELS.size())
    print(TRAIN_LABELS[0:5])
    
    TEST_FEATURES = torch.empty(1000, 2).uniform_(0, 1)
    TEST_LABELS   = h.get_labels(TEST_FEATURES, torch.empty(1, 2).fill_(0.5)) 
    
    
    plt.figure()
    h.plot_points(TRAIN_FEATURES, TRAIN_LABELS, "Training points and labels")
  
    model = Module.Sequential(Module.Linear(2, 25),
                              Module.ReLU(),
                              Module.Linear(25, 25),
                              Module.ReLU(),
                              Module.Linear(25, 2))


    loss_ce = Loss.CrossEntropy();
    opt = Optimizer.SGD(lr=0.05)
    
    model.train(TRAIN_FEATURES, TRAIN_LABELS, epochs=100, batch_size=10, opt=opt, loss=loss_ce)
    
    test_labels = model.predict(TEST_FEATURES, TEST_LABELS)

    h.plot_points(TEST_FEATURES, test_labels, "Test points and predictions")
    
test_softmax()
#test_sequential()

#test_sequential()