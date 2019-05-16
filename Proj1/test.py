from torch.autograd import Variable

from dlc_practical_prologue import generate_pair_sets
from models import MLPNet, ConvNet, DeepSiameseNet, BlockConvNet
from utilities import train_model, get_accuracy


def perform_experiments(n_runs=10, n_points=1000, n_epochs=200, run_best=False, verbose=False):
    """
    Perform experiments for 5 different neural network architectures and losses.
    
    To run all experiments call this function without default params
    
    :param n_runs: number of runs for which experiment should be repeated
    :param n_points: number of training and testing data points used in the experiments
    :param n_epochs: number of epochs every architecture should be trained on
    :param run_best: If True only the best architecture (Siamese Network with auxiliary loss) is trained
    :param verbose: If True, print training and validation loss every epoch
    :returns: dictionary containing history of training (training, validation loss and accuracy)
    """
    history_mlp_net = []
    history_conv_net = []
    history_conv_net_aux = []
    history_siamese = []
    history_siamese_aux = []
    
    for n_run in range(n_runs):
        data_set = generate_pair_sets(n_points)
        MAX_VAL = 255.0
        
        
        TRAIN_INPUT = Variable(data_set[0])/MAX_VAL 
        TRAIN_TARGET = Variable(data_set[1])
        TRAIN_CLASSES = Variable(data_set[2])
        
        TEST_INPUT = Variable(data_set[3])/MAX_VAL
        TEST_TARGET = Variable(data_set[4])
        TEST_CLASSES = Variable(data_set[5])
        
        if not run_best:
            ##############################################################################
            mlp_net = MLPNet(in_features=392, out_features=2, n_layers=3, n_hidden=16)
            mlp_net.train()
            history_mlp_net.append(train_model(mlp_net, train_input=TRAIN_INPUT.view((n_points, -1)), train_target=TRAIN_TARGET, aux_param=1.0,
                                               val_input=TEST_INPUT.view((n_points, -1)), val_target=TEST_TARGET, n_epochs=n_epochs, verbose=verbose))
            
            mlp_net.eval()
            acc = get_accuracy(mlp_net, TEST_INPUT.view((n_points, -1)), TEST_TARGET)*100.0
            print("Run: {}, Mlp_net Test Accuracy: {:.3f} %".format(n_run, acc))
            ##############################################################################
            conv_net = ConvNet(n_classes=2, n_layers=3, n_features=16)
            conv_net.train()
            history_conv_net.append(train_model(conv_net, train_input=TRAIN_INPUT, train_target=TRAIN_TARGET, aux_param=1.0,
                                                val_input=TEST_INPUT, val_target=TEST_TARGET, n_epochs=n_epochs, verbose=verbose))
            
            conv_net.eval()
            acc = get_accuracy(conv_net, TEST_INPUT, TEST_TARGET)*100.0
            print("Run: {}, ConvNet Test Accuracy: {:.3f} %".format(n_run, acc))
            ##############################################################################
            conv_net_aux = ConvNet(n_classes=22, n_layers=3, n_features=16)
            conv_net_aux.train()
            history_conv_net_aux.append(train_model(conv_net_aux, train_input=TRAIN_INPUT, train_target=TRAIN_TARGET, aux_param=1.0,
                                                    train_classes=TRAIN_CLASSES, val_input=TEST_INPUT, val_target=TEST_TARGET, val_classes=TEST_CLASSES,
                                                    n_epochs=n_epochs, verbose=verbose))
            
            conv_net_aux.eval()
            acc = get_accuracy(conv_net_aux, TEST_INPUT, TEST_TARGET)*100.0
            print("Run: {}, ConvNet Auxilary Test Accuracy: {:.3f} %".format(n_run, acc))
            ##############################################################################
            conv_net = BlockConvNet()
            conv_net_siamese = DeepSiameseNet(conv_net)
            
            conv_net.train()
            conv_net_siamese.train()
            history_siamese.append(train_model(conv_net_siamese, train_input=TRAIN_INPUT, train_target=TRAIN_TARGET,
                                               val_input=TEST_INPUT, val_target=TEST_TARGET, n_epochs=n_epochs, verbose=verbose))
            
            conv_net.eval()
            conv_net_siamese.eval()
            
            acc = get_accuracy(conv_net_siamese, TEST_INPUT, TEST_TARGET)*100.0
            print("Run: {}, Siamese Test Accuracy: {:.3f} %".format(n_run, acc))
        
        ##############################################################################
        conv_net = BlockConvNet()
        conv_net_siamese_aux = DeepSiameseNet(conv_net)
        
        conv_net.train()
        conv_net_siamese_aux.train()
        history_siamese_aux.append(train_model(conv_net_siamese_aux, train_input=TRAIN_INPUT, train_target=TRAIN_TARGET, train_classes=TRAIN_CLASSES,
                                               val_input=TEST_INPUT, val_target=TEST_TARGET, val_classes=TEST_CLASSES, aux_param=3.0,
                                               n_epochs=n_epochs, verbose=verbose))
        
        conv_net.eval()
        conv_net_siamese_aux.eval()
        
        acc = get_accuracy(conv_net_siamese_aux, TEST_INPUT, TEST_TARGET)*100.0
        print("Run: {}, Siamese Auxilary Test Accuracy: {:.3f} %".format(n_run, acc))
        ##############################################################################
        
        return {
                'history_mlp_net': history_mlp_net,
                'history_conv_net': history_conv_net,
                'history_conv_net_aux': history_conv_net_aux,
                'history_siamese': history_siamese,
                'history_siamese_aux':history_siamese_aux
                }
        
perform_experiments(n_runs=1, n_points=1000, run_best=True, verbose=True)