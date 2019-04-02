import torch

from strawberry import Module
from strawberry import Loss
from strawberry import Optimizer

# testing linear module
# training data
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