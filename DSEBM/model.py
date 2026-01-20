import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

class DSEBM(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DSEBM, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(self.state_dim + self.action_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.softp = nn.Softplus()

        self.bias_inv_1 = Parameter(torch.Tensor(self.hidden_dim))
        self.bias_inv_2 = Parameter(torch.Tensor(self.state_dim + self.action_dim))

        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        self.fc1.bias.data.zero_()
        self.fc2.bias.data.zero_()
        self.bias_inv_1.data.zero_()
        self.bias_inv_2.data.zero_()

    def random_noise(self, x):
        return torch.normal(mean = 0., std=1., size = x.shape).float()
    
    def forward(self, input):
        out = self.softp(self.fc1(input))
        out = self.softp(self.fc2(out))

        out = self.softp((out @ self.fc2.weight) + self.bias_inv_1)
        out = self.softp((out @ self.fc1.weight) + self.bias_inv_2)

        return out