import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np  
    
# data를 새롭게 representation하기 위한 AutoEncoder
class C_AutoEncoder(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, z_dim=4):
        super(C_AutoEncoder, self).__init__()
        self.z_dim = z_dim

        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_dim, eps=1e-04, affine=False)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.bn2 = nn.BatchNorm1d(hidden_dim, eps=1e-04, affine=False)
        self.fc3 = nn.Linear(hidden_dim, z_dim, bias=False)

        self.fc4 = nn.Linear(z_dim, hidden_dim, bias=False)
        self.bn3 = nn.BatchNorm1d(hidden_dim, eps=1e-4, affine = False)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.bn4 = nn.BatchNorm1d(hidden_dim, eps=1e-4, affine = False)
        self.fc6 = nn.Linear(hidden_dim, state_dim + action_dim, bias = False)
        
    def encoder(self, x):
        
        # encoder 구조는 Deep SVDD와 완전히 동일한 구조를 가지고 있음
        x = self.fc1(x)
        x = F.leaky_relu(self.bn1(x))
        x = self.fc2(x)
        x = F.leaky_relu(self.bn2(x))
        return self.fc3(x)
   
    def decoder(self, x):
        x = self.fc4(x)
        x = F.leaky_relu(self.bn3(x))
        x = self.fc5(x)
        x = F.leaky_relu(self.bn4(x))
        return self.fc6(x)
        
    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat
    
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 and classname != 'Conv':
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)

        
        
class network(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, z_dim=4):
        super(network, self).__init__()

        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_dim, eps=1e-04, affine=False)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.bn2 = nn.BatchNorm1d(hidden_dim, eps=1e-04, affine=False)
        self.fc3 = nn.Linear(hidden_dim, z_dim, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(self.bn1(x))
        x = self.fc2(x)
        x = F.leaky_relu(self.bn2(x))
        return self.fc3(x)