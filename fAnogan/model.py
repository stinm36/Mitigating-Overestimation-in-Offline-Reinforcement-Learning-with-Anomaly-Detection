import torch.nn as nn
import torch
import numpy as np

class Generator(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim=100):
        '''
        latent_dim: Latent vector dimension
        num_gf: Number of Generator Filters
        channels: Number of Generator output channels
        '''
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128, bias = False)
        )

        self.gen = nn.Sequential(
        nn.Linear(128, 256, bias=False),
        nn.BatchNorm1d(256, eps=1e-4, affine = False),
        nn.ReLU(),
        nn.Linear(256, 128, bias=False),
        nn.BatchNorm1d(128, eps=1e-4, affine = False),
        nn.ReLU(),
        nn.Linear(128, state_dim + action_dim, bias = False),
        )

    def forward(self, z):
        z = self.fc(z)
        z = self.gen(z)
        return z


class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Discriminator, self).__init__()

        self.encode = nn.Sequential(
        nn.Linear(state_dim + action_dim, hidden_dim, bias=False),
        nn.BatchNorm1d(hidden_dim, eps=1e-04, affine=False),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim, bias=False),
        nn.BatchNorm1d(hidden_dim, eps=1e-04, affine=False),
        nn.ReLU()
        )

        self.last_layer = nn.Sequential(
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        features = self.forward_features(x)
        out = self.last_layer(features)
        return out
    
    def forward_features(self, x):
        features = self.encode(x)
        return features
    
class Encoder(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, hidden_dim = 256):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim, eps=1e-4, affine = False),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim, eps=1e-4, affine = False),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim, bias=False)
        )

    def forward(self, x):
        return self.model(x)


# weight 초기화
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 and classname != 'Conv':
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)