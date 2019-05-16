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

    model = Module.Sequential(Module.Linear(in_features=6, out_features=1))
    loss_fun = Loss.MSE()
    opt = Optimizer.SGD(lr=0.01)
    
    model.train(TRAIN_FEATURES, TRAIN_LABELS, epochs=100, verbose=False,
                loss=loss_fun, opt=opt, batch_size=10)
    
    print("True weights: {}, {}".format(W, B))
    print("Learned weights: {}, {}".format(model.param()[0][0], 
          model.param()[1][0]))



def test_sequential(opt=Optimizer.ADAM(lr=0.001)):
    
    TRAIN_FEATURES = torch.empty(1000, 2).uniform_(0, 1)
    TRAIN_LABELS = h.get_labels(TRAIN_FEATURES, torch.empty(1, 2).fill_(0.5))
    
    TEST_FEATURES = torch.empty(1000, 2).uniform_(0, 1)
    TEST_LABELS   = h.get_labels(TEST_FEATURES, torch.empty(1, 2).fill_(0.5)) 

    
    plt.figure()
    h.plot_points(TRAIN_FEATURES, TRAIN_LABELS, "Training Data with Labels")
    
    model = Module.Sequential(Module.Linear(2, 25),
                              Module.ReLU(),
                              #Module.Dropout(p=.5),
                              Module.Linear(25, 25),
                              Module.ReLU(),
                              #Module.Dropout(p=.5),
                              Module.Linear(25, 2),
                              Module.Tanh())

    loss_fun = Loss.MSE()
    
    #opt = Optimizer.ADAM(lr=0.001)
    
    model.train(TRAIN_FEATURES, TRAIN_LABELS, epochs=200, batch_size=10, opt=opt, 
                loss=loss_fun, verbose=True, accuracy=True, 
                validation_set=(TEST_FEATURES, TEST_LABELS))

    test_labels = model.predict(TEST_FEATURES)
    
    plt.figure()
    h.plot_points(TEST_FEATURES, test_labels, "Test points and predictions")
    
    return model
    

def get_activation(ReLU):
    
    if ReLU:
        return Module.ReLU()
    else:
        return Module.Tanh()
    
    
def test_softmax(dropout=False, opt=Optimizer.ADAM(lr=0.001), ReLu=True, nb_neuron=25):
    
    TRAIN_FEATURES = torch.empty(1000, 2).uniform_(0, 1)
    TRAIN_LABELS = h.get_labels(TRAIN_FEATURES, torch.empty(1, 2).fill_(0.5))
    
    TEST_FEATURES = torch.empty(1000, 2).uniform_(0, 1)
    TEST_LABELS   = h.get_labels(TEST_FEATURES, torch.empty(1, 2).fill_(0.5)) 
    
    
    plt.figure()
    h.plot_points(TRAIN_FEATURES, TRAIN_LABELS, "Training points and labels")
  
    
    model = Module.Sequential()
    
    #model = Module.Sequential(Module.Linear(2, 25),
    #                          Module.ReLU(),
    #                          #Module.Dropout(p=.5),
    #                          Module.Linear(25, 25),
    #                          Module.ReLU(),
    #                          #Module.Dropout(p=.5),
    #                         Module.Linear(25, 2))


    model.add(Module.Linear(2, nb_neuron))
    model.add(get_activation(ReLu))
      

    if dropout:
        model.add(Module.Dropout(p=.5))
        
    model.add(Module.Linear(nb_neuron, nb_neuron))
    model.add(get_activation(ReLu))
    
    if dropout:
        model.add(Module.Dropout(p=.5))
        
    model.add(Module.Linear(nb_neuron, 2))

    loss_ce = Loss.CrossEntropy();
    
    model.train(TRAIN_FEATURES, TRAIN_LABELS, epochs=200, batch_size=10, opt=opt, 
                loss=loss_ce, verbose=True, accuracy=True, 
                validation_set=(TEST_FEATURES, TEST_LABELS))
                    
    test_labels = model.predict(TEST_FEATURES)

    h.plot_points(TEST_FEATURES, test_labels, "Test points and predictions")
    return model
    
#test linear model 
#test_linear()


#MSE_model = test_sequential()
#Cross_entropy_model = test_softmax()

"""fig = h.comparison_plot(MSE_model.history,
                        Cross_entropy_model.history,
                        label1="MSE",
                        label2="Cross Entropy",
                        title="MSE vs Cross Entropy",
                        fig_size=(5, 15))
"""

#ReLu_model = test_softmax()
#Tanh_model = test_softmax(ReLu=False)

"""fig = h.comparison_plot(ReLu_model.history,
                        Tanh_model.history,
                        label1="ReLU",
                        label2="Tanh",
                        title="Neural Network with ReLU vs Tanh",
                        fig_size=(5, 15))
"""

Nodrop_model  = test_softmax(nb_neuron=500)
Yesdrop_model = test_softmax(nb_neuron=500, dropout=True)

fig = h.comparison_plot(Nodrop_model.history,
                        Yesdrop_model.history,
                        label1="No Dropout",
                        label2="Dropout",
                        title="Neural Network with and without dropouts",
                        fig_size=(3, 8))

