import numpy as np
import os
import argparse
import gym
import d4rl
import datetime
import torch

from fAnogan.dataset import D4RLDataset
from fAnogan.model import Generator, Discriminator, Encoder

from torch.utils.data import DataLoader

from fAnogan.train_wgan import train_wgangp
from fAnogan.train_encoder_izif import train_encoder_izif

# https://github.com/A03ki/f-AnoGAN

def experiment(args):
    if torch.cuda.is_available():
        print("GPU IS AVAILABLE!")
    device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cpu")
    
    env = gym.make(args.env)
    args.obs_dim = obs_dim = env.observation_space.low.size
    args.action_dim = action_dim = env.action_space.low.size

    dataset = D4RLDataset(args.env)
    data_loader = DataLoader(dataset, batch_size=args.batch_size_ad, shuffle=True, num_workers = 4, pin_memory = True, drop_last=True)

    generator = Generator(args.obs_dim, args.action_dim, latent_dim=args.fanogan_latent_dim)
    discriminator = Discriminator(args.obs_dim, args.action_dim, hidden_dim=args.hidden_dim)
    encoder = Encoder(args.obs_dim, args.action_dim, latent_dim=args.fanogan_latent_dim, hidden_dim=args.hidden_dim)

    train_wgangp(args, generator, discriminator,
                 data_loader, device, lambda_gp=10)
    
    train_encoder_izif(args, generator, discriminator, encoder,
                       data_loader, device, kappa=1.0)

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
    parser = argparse.ArgumentParser(description='f-AnoGAN')

    # From BEAR
    parser.add_argument("--env", type=str, default='hopper-medium-v2')
    parser.add_argument("--algo_name", type=str, default='f-AnoGAN')
    parser.add_argument("--gpu", default='0', type=str)
    parser.add_argument('--seed', default= int(np.random.randint(0, 100000)), type=int)
    parser.add_argument('--log_dir', default='./default/', type=str, 
                        help="Location for logging")
    parser.add_argument('--epochs_ad', default=200, type=int)
    parser.add_argument('--lr_ad', default=1e-3, type=float)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--ad_train', default=True, type=str2bool)
    parser.add_argument('--latent_dim', default=128, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--normal_class', default=1, type=int)
    parser.add_argument('--batch_size_ad', default=256, type=int)
    parser.add_argument('--ad_module', default="fanogan", type=str)
    parser.add_argument('--ad_save_path', default="./weights", type=str)
    
    # f-AnoGAN hyperparameters
    parser.add_argument('--fanogan_latent_dim', default=100, type=int)
    parser.add_argument('--fanogan_n_critic', default=5, type=int)

    args = parser.parse_args()
    
    args.log_dir = os.path.join(args.log_dir, args.ad_module)
    args.ad_save_path = os.path.join(args.ad_save_path, args.ad_module)
    
    experiment(args)