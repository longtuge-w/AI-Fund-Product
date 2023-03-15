import torch
from torch import nn, optim
import torch.nn.functional as F

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        m.weight.data.normal_(mean = 0.0, std = 0.05)
        m.bias.data.fill_(0.05)


class CMLE(nn.Module):
    def __init__(self, n_features):
        super(CMLE, self).__init__()
        self.n_features = n_features
        self.linear1 = nn.Linear(self.n_features, self.n_features * 4)
        self.linear2 = nn.Linear(self.n_features * 4, self.n_features * 2)
        self.linear3 = nn.Linear(self.n_features * 2, self.n_features // 2)
        self.linear4 = nn.Linear(self.n_features // 2, 1)
        self.apply(weights_init)
    
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        result = F.relu(self.linear4(x))
        result = result.view(result.shape[0], result.shape[1])
        return result