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
    
def prep_dataloader(env_id="hopper-medium-v2", batch_size=256, seed=1):
    env = gym.make(env_id)
    dataset = env.get_dataset()
    tensors = {}
    for k, v in dataset.items():
        if k in ["actions", "observations", "next_observations", "rewards", "terminals"]:
            if  k is not "terminals":
                tensors[k] = torch.from_numpy(v).float()
            else:
                tensors[k] = torch.from_numpy(v).long()

    tensordata = TensorDataset(tensors["observations"],
                               tensors["actions"],
                               tensors["rewards"][:, None],
                               tensors["next_observations"],
                               tensors["terminals"][:, None])
    dataloader  = DataLoader(tensordata, batch_size=batch_size, shuffle=True)
    
    if "hopper" in env_id:
        eval_env = gym.make("Hopper-v2")
        eval_env = gym.wrappers.NormalizeReward(eval_env)
    eval_env.seed(seed)
    return dataloader, eval_env