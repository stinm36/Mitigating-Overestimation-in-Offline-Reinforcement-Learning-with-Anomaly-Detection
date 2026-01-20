import os
import argparse
from solver import Solver
from torch.backends import cudnn
from utils import *
from dataset import *
from torch.utils.data import DataLoader
import gym

def str2bool(v):
    return v.lower() in ('true')

def main(config):
    # For fast training
    cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu)

    # Create directories if not exist
    mkdir(os.path.join(config.log_path, config.env))
    mkdir(os.path.join(config.model_save_path, config.env))
    
    config.log_path = os.path.join(config.log_path, config.env)
    config.model_save_path = os.path.join(config.model_save_path, config.env)

    dataset = D4RLDataset(config.env)
    data_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    env = gym.make(config.env)
    
    obs_dim = env.observation_space.low.size
    action_dim = env.action_space.low.size
    
    input_dim = obs_dim + action_dim
    
    # Solver
    solver = Solver(input_dim, data_loader, config.latent_dim, vars(config))
    

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()

    return solver
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--lr', type=float, default=5e-4)


    # Training settings
    parser.add_argument('--gpu', type=str, default=0)
    parser.add_argument('--env', type=str, default='halfcheetah-medium-v2')
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--gmm_k', type=int, default=4)
    parser.add_argument('--lambda_energy', type=float, default=0.1)
    parser.add_argument('--lambda_cov_diag', type=float, default=0.005)
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--latent_dim', type=int, default=2)

    # Misc
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)

    # Path
    parser.add_argument('--log_path', type=str, default='./dagmm/logs')
    parser.add_argument('--model_save_path', type=str, default='./dagmm/models')

    # Step size
    parser.add_argument('--log_step', type=int, default=61)
    parser.add_argument('--sample_step', type=int, default=61)
    parser.add_argument('--model_save_step', type=int, default=61)

    config = parser.parse_args()
 
    args = vars(config)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')

    main(config)