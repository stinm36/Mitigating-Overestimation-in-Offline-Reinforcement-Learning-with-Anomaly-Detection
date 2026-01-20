import numpy as np
import torch
from torch.utils.data import DataLoader
import os
import argparse
import gym
import d4rl
import datetime

from gym.envs.mujoco import HalfCheetahEnv
import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.rlocc import SACTrainer
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

from svdd import *
from svdd.svdd import *

from dagmm.model import *
from dagmm.solver import Solver
from dagmm.utils import *

from fAnogan.model import Generator, Discriminator, Encoder
from fAnogan.dataset import *
from fAnogan.train_encoder_izif import *
from fAnogan.train_wgan import *
from fAnogan.trainer import *

from DSEBM.model import *
from DSEBM.trainer import Trainer as DSEBMTrainer

from partial_dataset import *

def experiment(args, variant):
    if torch.cuda.is_available():
        print("GPU IS AVAILABLE!")
    device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cpu")

    eval_env = gym.make(variant['env_name'])
    expl_env = eval_env
    
    args.obs_dim = obs_dim = expl_env.observation_space.low.size
    args.action_dim = action_dim = eval_env.action_space.low.size

    dataset = D4RLDataset(args.env, fraction=args.fraction, seed=args.seed, preserve_ratio=args.preserve_ratio)
    dataloader= DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers = 4, pin_memory = True, drop_last = True)
    

    if args.ad_module == 'svdd':
        ad = TrainerDeepSVDD(args, obs_dim, action_dim, dataloader, device)

        if args.ad_train:
            ad.pretrain()
            ad.train()

        ad = ad.load_SVDD()
    elif args.ad_module == 'dagmm':
        # load 추가
        ad = Solver(obs_dim+action_dim, dataloader, args, device)

        if args.ad_train:
            ad.train()

        ad.load_pretrained_model()
    elif args.ad_module == 'dsebm':
        args.lr = args.lr_ad
        args.dim = obs_dim + action_dim
        ad_model = DSEBM(obs_dim, action_dim, hidden_dim = args.hidden_dim).to(device)
        ad = DSEBMTrainer(ad_model, device, dataloader, args)

        if args.ad_train:
            ad.train()

        ad.load_ckpt()
    elif args.ad_module == 'fanogan':
        generator = Generator(args.obs_dim, args.action_dim, latent_dim=args.fanogan_latent_dim)
        discriminator = Discriminator(args.obs_dim, args.action_dim, hidden_dim=args.hidden_dim)
        encoder = Encoder(args.obs_dim, args.action_dim, latent_dim=args.fanogan_latent_dim, hidden_dim=args.hidden_dim)

        if args.ad_train:
            train_wgangp(args, generator, discriminator,
                 dataloader, device, lambda_gp=10)
    
            train_encoder_izif(args, generator, discriminator, encoder,
                            dataloader, device, kappa=1.0)
            
        statedict = torch.load(os.path.join(args.ad_save_path, 'fanogan', args.env, 'gan.pth'))
        statedict_encoder = torch.load(os.path.join(args.ad_save_path, 'fanogan', args.env, 'encoder.pth'))

        generator.load_state_dict(statedict['Generator'])
        discriminator.load_state_dict(statedict['Discriminator'])
        encoder.load_state_dict(statedict_encoder)
        
        generator.to(device).eval()
        discriminator.to(device).eval()
        encoder.to(device).eval()

        ad = {'Generator' : generator, 'Discriminator' : discriminator, 'Encoder' : encoder, 'kappa' : args.kappa}
    else:
        raise Exception("Wrong Module Name")


    M = variant['layer_size']
    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M],
    )
    eval_policy = MakeDeterministic(policy)
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        policy,
    )
    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )
    subsample_idx = prefill_rlkit_replay_buffer_from_d4rl(
    replay_buffer=replay_buffer,
    env_name=args.env,
    fraction=args.fraction,
    seed=args.seed,
    preserve_ratio=args.preserve_ratio,
    done_includes_timeouts=args.done_includes_timeouts,
    verbose=True,
    )
    print("can sample:", replay_buffer.num_steps_can_sample())
    trainer = SACTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        ad = ad,
        args = args,
        weight_function=args.weight_function,
        **variant['trainer_kwargs']
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()


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
    parser = argparse.ArgumentParser(description='RLOCC')

    # Common args
    parser.add_argument("--env", type=str, default='hopper-medium-v2')
    parser.add_argument("--algo_name", type=str, default='RLOCC')
    parser.add_argument("--gpu", default='0', type=str)
    parser.add_argument('--seed', default= int(np.random.randint(0, 100000)), type=int)
    parser.add_argument('--all_saves', default="saves", type=str)
    parser.add_argument('--trial_name', default="", type=str)
    parser.add_argument('--log_dir', default='./default/', type=str, 
                        help="Location for logging")
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--fraction', default=0.10, type=float)
    parser.add_argument('--preserve_ratio', default=True, type=str2bool)
    parser.add_argument('--done_includes_timeouts', default=False, type=str2bool)

    # For agent
    parser.add_argument('--qf_lr', default=3e-4, type=float)
    parser.add_argument('--policy_lr', default=3e-4, type=float)
    parser.add_argument('--nepochs', default=3000, type=int)

    parser.add_argument('--theoretically', default=3, type=int)
    parser.add_argument('--weight_function', default='identity', type=str)

    # Common for AD models
    parser.add_argument('--epochs_ad', default=100, type=int)
    parser.add_argument('--patience', default=50, type=int)
    parser.add_argument('--lr_ad', default=1e-4, type=float)
    parser.add_argument('--lr_milestones', default=[50, 150], type=list)
    parser.add_argument('--ad_train', default=True, type=str2bool)
    parser.add_argument('--latent_dim', default=128, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--normal_class', default=1, type=int)

    # SVDD
    parser.add_argument('--weight_decay_svdd', default=0.5e-6, type=float)
    parser.add_argument('--weight_decay_ae', default=0.5e-3, type=float)
    parser.add_argument('--ad_module', default="svdd", type=str)
    parser.add_argument('--ad_save_path', default="./weights_ad", type=str)

    # DAGMM hyperparameters
    parser.add_argument('--gmm_k', type=int, default=2)
    parser.add_argument('--lambda_energy', type=float, default=0.1)
    parser.add_argument('--lambda_cov_diag', type=float, default=0.005)
    parser.add_argument('--pretrained_model', default=None)
    parser.add_argument('--dagmm_latent_dim', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)
    parser.add_argument('--log_step', type=int, default=500)
    parser.add_argument('--sample_step', type=int, default=500)
    parser.add_argument('--model_save_step', type=int, default=500)
    
    # DSEBM hyperparameters
    # parser.add_argument('--dim', default=1, type=int)

    # f-AnoGAN hyperparameters
    parser.add_argument('--fanogan_latent_dim', default=100, type=int)
    parser.add_argument('--fanogan_n_critic', default=5, type=int)
    parser.add_argument('--kappa', default=1.0, type=float)

    args = parser.parse_args()

    args.ad_save_path = args.ad_save_path + '_' + str(args.fraction)
    
    # noinspection PyTypeChecker
    variant = dict(
        algorithm="RLAD",
        env_name=args.env,
        version="normal",
        layer_size=256,
        replay_buffer_size=int(1E6),
        algorithm_kwargs=dict(
            num_epochs=args.nepochs,
            num_eval_steps_per_epoch=5000,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=1000,
            max_path_length=1000,
            batch_size=args.batch_size,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=args.policy_lr,
            qf_lr=args.qf_lr,
            reward_scale=1,
            use_automatic_entropy_tuning=True,
        ),
    )
    os.makedirs(os.path.join(args.ad_save_path, args.ad_module, args.env), exist_ok=True)
    file_name = args.trial_name
    setup_logger(file_name, variant=variant, base_log_dir=args.all_saves, name = file_name, log_dir=os.path.join(args.all_saves ,str(args.weight_function), args.log_dir, file_name, args.ad_module ,args.env,'%s'%(datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))))
    ptu.set_gpu_mode(True)
    experiment(args, variant)