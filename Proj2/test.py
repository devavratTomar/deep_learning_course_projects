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
    """
    Testes Linear Layer with dummy prediction problem
    """
    W = torch.tensor([[5], [1], [-6], [7], [8], [-2]], dtype=torch.float)
    B = -3
    
    # create dummy data with noisy labels
    TRAIN_FEATURES = torch.empty(1000, 6).uniform_(-1, 1)
    
    # compute labels as Wx+b plus some gaussian noise
    TRAIN_LABELS = torch.mm(TRAIN_FEATURES, W) + B + torch.empty((1000, 1)).normal_(0, 0.5)

    # build a sequential model with 6 input features and one output
    model = Module.Sequential(Module.Linear(in_features=6, out_features=1))
    
    # use MSE as a kiss function
    loss_fun = Loss.MSE()
    
    #use SGD optimizer
    opt = Optimizer.SGD(lr=0.01)
    
    # train our model given our data for 100 epochs with 10 batch size
    model.train(TRAIN_FEATURES, TRAIN_LABELS, epochs=100, verbose=False,
                loss=loss_fun, opt=opt, batch_size=10)
    
    print("True weights: {}, {}".format(W, B)) #true weights without noise
    print("Learned weights: {}, {}".format(model.param()[0][0], 
          model.param()[1][0]))



def test_sequential(opt, epochs=200):
    """
    This method tests sequential method using MSE loss and takes optimizer as a parameter
    """
    
    # train data
    TRAIN_FEATURES = torch.empty(1000, 2).uniform_(0, 1)
    # get labels depending if in or outside of the circle
    TRAIN_LABELS = h.get_labels(TRAIN_FEATURES, torch.empty(1, 2).fill_(0.5))
    
    # test data
    TEST_FEATURES = torch.empty(1000, 2).uniform_(0, 1)
    # get labels for the test set
    TEST_LABELS   = h.get_labels(TEST_FEATURES, torch.empty(1, 2).fill_(0.5)) 

    
    plt.figure()
    h.plot_points(TRAIN_FEATURES, TRAIN_LABELS, "Training Data with Labels")
    
    # create a sequential model with 3 Linear layers, firts 2 with Relu and tha last with
    # Tanh activation function
    model = Module.Sequential(Module.Linear(2, 25),
                              Module.ReLU(),
                              Module.Linear(25, 25),
                              Module.ReLU(),
                              Module.Linear(25, 25),
                              Module.ReLU(),
                              Module.Linear(25, 2),
                              Module.Tanh())

    # use MSE Loss
    loss_fun = Loss.MSE()
        
    # train model fr 200 epochs for the given optimizer with batch size of 10
    model.train(TRAIN_FEATURES, TRAIN_LABELS, epochs=epochs, batch_size=10, opt=opt, 
                loss=loss_fun, verbose=True, accuracy=True, 
                validation_set=(TEST_FEATURES, TEST_LABELS))

    
    # predict labels on a test set
    test_labels = model.predict(TEST_FEATURES)
    
    plt.figure()
    h.plot_points(TEST_FEATURES, test_labels, "Test points and predictions")
    
    return model
    

def get_activation(ReLU):
    """
    Returns a new ReLU module if parameter is true, otherwise Tanh
    """
    if ReLU:
        return Module.ReLU()
    else:
        return Module.Tanh()
    
    
def test_softmax(opt, dropout=False, ReLu=True, nb_neuron=25, epochs=200):
    """
    Learns sequential model with softmax layer and cross entropy loss
    
    :param opt:        optimizer to be used
    :param dropout:    whether to use dropouts
    :param ReLu:       whether to use ReLU (otherwise Tanh)
    :param nb_neuron:  number of neurons for hidden layer
        
    :return:           trained model
    """
    
    # train data
    TRAIN_FEATURES = torch.empty(1000, 2).uniform_(0, 1)
    # get training labels depending if in or outside of the circle
    TRAIN_LABELS = h.get_labels(TRAIN_FEATURES, torch.empty(1, 2).fill_(0.5))
    
    # test data
    TEST_FEATURES = torch.empty(1000, 2).uniform_(0, 1)
    # get labels for the test set
    TEST_LABELS   = h.get_labels(TEST_FEATURES, torch.empty(1, 2).fill_(0.5)) 
    
    
    plt.figure()
    h.plot_points(TRAIN_FEATURES, TRAIN_LABELS, "Training points and labels")
  
    # create a sequential model
    model = Module.Sequential()

    # add 1st linear layer with ReLu or Tanh activation depending on a ReLU parameter
    model.add(Module.Linear(2, nb_neuron))
    model.add(get_activation(ReLu))
      

    if dropout:
        # add dropouts in case dropout=True with p value of 0.5
        model.add(Module.Dropout(p=.5))
      
    # add hidden layer with nb_neuron neurons followed by an activation
    model.add(Module.Linear(nb_neuron, nb_neuron))
    model.add(get_activation(ReLu))
    
    if dropout:
        model.add(Module.Dropout(p=.5))
        
    # add hidden layer with nb_neuron neurons followed by an activation
    model.add(Module.Linear(nb_neuron, nb_neuron))
    model.add(get_activation(ReLu))
    
    if dropout:
        model.add(Module.Dropout(p=.5))
        
    # add final linear layer
    model.add(Module.Linear(nb_neuron, 2))

    # crossentropy loss combines with the final softmax layer
    loss_ce = Loss.CrossEntropy();
    
    # train the model for the given parameters + 200 epochs, 10 batch size
    model.train(TRAIN_FEATURES, TRAIN_LABELS, epochs=epochs, batch_size=10, opt=opt, 
                loss=loss_ce, verbose=True, accuracy=True, 
                validation_set=(TEST_FEATURES, TEST_LABELS))
     
    # predict outcome of the test data               
    test_labels = model.predict(TEST_FEATURES)

    h.plot_points(TEST_FEATURES, test_labels, "Test points and predictions")
    return model



def run_experiment(epochs=200):
   
    # Relu and Tanh comparison + plot
    print("Training Sequential NN with ReLU as activation function")
    ReLu_model = test_softmax(opt=Optimizer.ADAM(lr=0.001),
                              epochs=epochs)
    print("Training Sequential NN with Tanh as activation function")
    Tanh_model = test_softmax(opt=Optimizer.ADAM(lr=0.001), 
                              ReLu=False,
                              epochs=epochs)
    
    fig = h.comparison_plot(ReLu_model.history,
                            Tanh_model.history,
                            label1="ReLU",
                            label2="Tanh",
                            title="Neural Network with ReLU vs Tanh",
                            fig_size=(4, 10))
    
    
    # with and without dropout comparison + plot
    # for report it was run using nb_neuron=500
    print("Training Sequential NN with CE loss and without dropouts")
    Nodrop_model  = test_softmax(opt=Optimizer.ADAM(lr=0.001), 
                                 epochs=epochs)
    print("Training Sequential NN with CE loss and dropouts(p=0.5)")
    Yesdrop_model = test_softmax(opt=Optimizer.ADAM(lr=0.001), 
                                 dropout=True,
                                 epochs=epochs)
    
    fig = h.comparison_plot(Nodrop_model.history,
                            Yesdrop_model.history,
                            label1="No Dropout",
                            label2="Dropout",
                            title="Neural Network with and without dropouts",
                            fig_size=(4, 10))
    
    #ADAM and SGD comparison + plot
    print("Training Sequential NN with SGD")
    SGD_model  = test_softmax(opt=Optimizer.SGD(lr=0.1),
                              epochs=epochs)
    print("Training Sequential NN with ADAM optimizer")
    ADAM_model = test_softmax(Optimizer.ADAM(lr=0.001),
                              epochs=epochs)
    
    fig = h.comparison_plot(SGD_model.history,
                            ADAM_model.history,
                            label1="SGD",
                            label2="ADAM",
                            title="Neural Network with SGD vs ADAM",
                            fig_size=(4, 10))
    
###########################################_______#########################################
##########################################| TEST |#########################################
##########################################|______|#########################################

#test linear model 
print("Training Linear Model")
test_linear()

#MSE and CrossEntropy comparison + plot
print("Training Sequential NN with MSE Loss")
MSE_model = test_sequential(opt=Optimizer.ADAM(lr=0.001),
                            epochs=100)
print("Training Sequential NN with Cross Entropy Loss")
Cross_entropy_model = test_softmax(opt=Optimizer.ADAM(lr=0.001),
                                   epochs=100)

fig = h.comparison_plot(MSE_model.history,
                        Cross_entropy_model.history,
                        label1="MSE",
                        label2="Cross Entropy",
                        title="MSE and Cross Entropy",
                        fig_size=(4, 10))



#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# The experiments take long, because they run all the possible combinations
# 6 models in total, uncomment the line below if you want to run it.
# In report plots were produced with default values, meaning epochs=200
# to make it faster this one will be run with only 100 epochs
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#run_experiment(epochs=100)