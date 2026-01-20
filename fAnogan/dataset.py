import gym
import d4rl
import torch

from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset
    
# D4RL 데이터셋 로드를 위한 클래스
class D4RLDataset(Dataset):
    def __init__(self, env_name):
        env = gym.make(env_name)
        dataset = d4rl.qlearning_dataset(env)
        self.observations = dataset['observations']
        self.action = dataset['actions']
        self.data = torch.cat([torch.tensor(self.observations, dtype=torch.float32), torch.tensor(self.action, dtype=torch.float32)], dim=1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]