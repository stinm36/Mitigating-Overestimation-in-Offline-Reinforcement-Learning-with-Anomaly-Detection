# https://github.com/intrudetection/robevalanodetect/tree/main

import torch
import numpy as np
import os
import argparse
import gym, d4rl
from model import DSEBM
from trainer import Trainer
from torch.utils.data import DataLoader
from dataset import D4RLDataset

def experiment(args):
    env = gym.make(args.env)

    if torch.cuda.is_available():
        print("GPU IS AVAILABLE!")
    device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cpu")
    
    obs_dim = env.observation_space.low.size
    action_dim = env.action_space.low.size

    args.dim = obs_dim + action_dim

    dataset = D4RLDataset(args.env)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers = 4, pin_memory = True, drop_last=True)

    model = DSEBM(obs_dim, action_dim, hidden_dim=args.hidden_dim).to(device)
    trainer = Trainer(model, device, dataloader, args)
    trainer.train()


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    # noinspection PyTypeChecker
    parser = argparse.ArgumentParser(description='DSEBM')

    # From BEAR
    parser.add_argument("--env", type=str, default='hopper-medium-v2')
    parser.add_argument("--gpu", default='0', type=str)
    parser.add_argument('--seed', default= int(np.random.randint(0, 100000)), type=int)

    parser.add_argument('--n_epochs', default=500, type=int)
    parser.add_argument('--patience', default=50, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--lr_milestones', default=[50, 200], type=list)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)

    args = parser.parse_args()
    
    experiment(args)
