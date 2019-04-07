from torch.autograd import Variable

from dlc_practical_prologue import generate_pair_sets
from models import MLPNet, ConvNet
from utilities import train_model, get_accuracy


data_set = generate_pair_sets(1000)

TRAIN_INPUT = Variable(data_set[0])
TRAIN_TARGET = Variable(data_set[1])
TRAIN_CLASSES = Variable(data_set[2])

TEST_INPUT = Variable(data_set[3])
TEST_TARGET = Variable(data_set[4])
TEST_CLASSES = Variable(data_set[5])


conv_net_1 = ConvNet(n_classes=22, n_layers=3, n_features=128)
train_model(conv_net_1, train_input=TRAIN_INPUT, train_target=TRAIN_TARGET, aux_param=1.0, train_classes=TRAIN_CLASSES)
print("Test Accuracy: {:.3f} %".format(get_accuracy(conv_net_1, TEST_INPUT, TEST_TARGET)*100.0))