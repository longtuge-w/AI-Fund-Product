from torch import nn


class DNN(nn.Module):
    def __init__(self, n_features, hidden_size, date_size):
        super(DNN, self).__init__()
        self.n_features = n_features
        self.linear1 = nn.Linear(n_features, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, 1)
        self.linear5 = nn.Linear(date_size, 1)
        self.act1 = nn.Sigmoid()
        self.act2 = nn.Sigmoid()
        self.act3 = nn.LeakyReLU()
        self.act4 = nn.LeakyReLU()
        self.norm1 = nn.BatchNorm1d(hidden_size)
        self.norm2 = nn.BatchNorm1d(hidden_size)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.act1(x)
        x = x.permute(0,2,1)
        x = self.norm1(x)
        x = x.permute(0,2,1)
        x = self.linear2(x)
        x = self.act2(x)
        x = x.permute(0,2,1)
        x = self.norm2(x)
        x = x.permute(0,2,1)
        x = self.linear3(x)
        x = self.act3(x)
        x = self.linear4(x)
        x = x.squeeze(-1)
        x = self.act4(x)
        x = self.linear5(x)
        x = x.squeeze(-1)
        return x