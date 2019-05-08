from torch.autograd import Variable

from dlc_practical_prologue import generate_pair_sets
from models import MLPNet, ConvNet, DeepSiameseNet, BlockConvNet
from utilities import train_model, get_accuracy


data_set = generate_pair_sets(1000)

TRAIN_INPUT = Variable(data_set[0])
TRAIN_TARGET = Variable(data_set[1])
TRAIN_CLASSES = Variable(data_set[2])

TEST_INPUT = Variable(data_set[3])
TEST_TARGET = Variable(data_set[4])
TEST_CLASSES = Variable(data_set[5])

#
#conv_net = ConvNet(n_classes=22, n_layers=3, n_features=128)
#train_model(conv_net, train_input=TRAIN_INPUT, train_target=TRAIN_TARGET, aux_param=1.0, train_classes=TRAIN_CLASSES)

conv_net = BlockConvNet()
conv_net_siamese = DeepSiameseNet(conv_net)

conv_net.train()
conv_net_siamese.train()
train_model(conv_net_siamese, train_input=TRAIN_INPUT, train_target=TRAIN_TARGET, n_epoch=100)

conv_net.eval()
conv_net_siamese.eval()
print("Test Accuracy: {:.3f} %".format(get_accuracy(conv_net_siamese, TEST_INPUT, TEST_TARGET)*100.0))