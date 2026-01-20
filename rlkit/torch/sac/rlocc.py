from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn
import torch.nn.functional as F

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.launchers.launcher_util import setup_logger
from rlkit.torch.torch_rl_algorithm import TorchTrainer

from svdd import *
from fAnogan.model import *
from fAnogan.train_encoder_izif import *
from fAnogan.train_wgan import *
from fAnogan.trainer import *
from DSEBM.model import *

def id(x):
    return x

def calc_anomaly_score(ad, state_action, c = None, ad_type = None, **kwargs):
    if ad_type == 'svdd':
        return torch.sqrt(torch.sum((ad(state_action)-c)**2, dim=tuple(range(1, state_action.dim()))))
    
    elif ad_type == 'dagmm':
        state_action = ad.to_var(state_action)
        _, _, z, _ = ad.dagmm(state_action)
        weight, _ = ad.dagmm.compute_energy(z, size_average=False)
        return weight
    
    elif ad_type == 'dsebm':
        out = ad.model(state_action)
        return ad.energy(state_action, out)
    
    elif ad_type == 'fanogan':
        criterion = nn.MSELoss()

        g = ad['Generator']
        d = ad['Discriminator']
        e = ad['Encoder']

        real_z = e(state_action)
        fake_data = g(real_z)
        fake_z = e(fake_data)

        real_features = d.forward_features(state_action)
        fake_features = d.forward_features(fake_data)

        data_distance = criterion(fake_data, state_action)
        loss_features = criterion(fake_features, real_features)
        anomaly_score = data_distance + ad['kappa']*loss_features
        return anomaly_score
    
    else:
        raise Exception("No Anomaly Module is selected")

class SACTrainer(TorchTrainer):
    def __init__(
            self,
            env,
            policy,
            qf1,
            qf2,
            target_qf1,
            target_qf2,
            ad,
            args,

            discount=0.99,
            reward_scale=1.0,

            policy_lr=1e-3,
            qf_lr=1e-3,
            optimizer_class=optim.Adam,

            soft_target_tau=1e-2,
            target_update_period=1,
            plotter=None,
            render_eval_paths=False,

            use_automatic_entropy_tuning=True,
            target_entropy=None,
            policy_eval_start=0,

            weight_function = 'sigmoid'
    ):
        super().__init__()
        self.env = env
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod(self.env.action_space.shape).item()  # heuristic value from Tuomas
            self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=policy_lr,
            )

        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
        )

        self.discount = discount
        self.reward_scale = reward_scale
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True
        self.discrete = False

        self.args = args

        self.c = None

        if args.ad_module == 'svdd':
            self.ad, self.c = ad
            self.ad.cuda().eval()
            self.c = self.c.cuda()
        elif args.ad_module == 'dagmm':
            self.ad = ad
            self.ad.dagmm.cuda().eval()
        elif args.ad_module == 'dsebm':
            self.ad = ad
            self.ad.model.cuda().eval()
        elif args.ad_module == 'fanogan':
            self.ad = ad
        else:
            raise Exception("Wrong Module Name")

        self.policy_eval_start = policy_eval_start

        self._current_epoch = 0
        self._policy_update_ctr = 0
        self._num_q_update_steps = 0
        self._num_policy_update_steps = 0
        self._num_policy_steps = 1

        self.theoretically = self.args.theoretically
        print(f'##########Theoretical Type is {self.theoretically}##########')

        if weight_function == 'sigmoid':
            def weight_sigmoid(w):
                w = w + 1e-4
                w.pow_(-1)
                w = torch.sigmoid(w)
                return w
            self.weight_function = weight_sigmoid
        elif weight_function == 'sigmoid_theoretically':
            def weight_sigmoid(w):
                w = w + 1e-4
                w.pow_(-1)
                w = torch.max(torch.sigmoid(w), torch.tensor(0.01, device = w.device, dtype=w.dtype))
                return w
            self.weight_function = weight_sigmoid
            self.theoretically = 2
        elif weight_function == 'identity':
            def weight_id(w):
                w = (w+1e-4).pow(-1)
                return w
            self.weight_function = weight_id
        # elif weight_function == 'hard_label':
        #     def weight_hard(w):
        #         w = torch.where(w > threshold, torch.tensor(0.).cuda(), torch.tensor(1.).cuda())
        #         return w
        #         # if w > threshold: # anomaly로 분류 -> update를 아예 안함
        #         #     return 0
        #         # else: # normal로 분류 -> update 온전히
        #         #     return 1
        #     self.weight_function = weight_hard
        elif weight_function == 'exponential':
            def weight_exp(w, beta = 1.0):
                return torch.exp(-beta*w)
            self.weight_function = weight_exp
        else:
            self.weight_function = weight_function
        
    def eval_q_custom(self, custom_policy, data_batch, q_function=None):
        if q_function is None:
            q_function = self.qf1
        
        obs = data_batch['observations']
        # Evaluate policy Loss
        new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy(
            obs, reparameterize=True, return_log_prob=True,
        )
        q_new_actions = q_function(obs, new_obs_actions)
        return float(q_new_actions.mean().detach().cpu().numpy())

    def train_from_torch(self, batch):
        self._current_epoch += 1
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        with torch.cuda.amp.autocast(enabled=False):
            """
            Policy and Alpha Loss
            """
            new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy(
                obs, reparameterize=True, return_log_prob=True,
            )
            if self.use_automatic_entropy_tuning:
                alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                alpha = self.log_alpha.exp()
            else:
                alpha_loss = 0
                alpha = 1
            
            q_new_actions = torch.min(
                self.qf1(obs, new_obs_actions),
                self.qf2(obs, new_obs_actions),
            )

            ## Policy update

            policy_loss = (alpha*log_pi - q_new_actions).mean()

            if self._current_epoch < self.policy_eval_start:
                """
                For the initial few epochs, try doing behaivoral cloning, if needed
                conventionally, there's not much difference in performance with having 20k 
                gradient steps here, or not having it
                """
                policy_log_prob = self.policy.log_prob(obs, actions)
                policy_loss = (alpha * log_pi - policy_log_prob).mean()
            
            self._num_policy_update_steps += 1
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            """
            QF Loss
            """
            q1_pred = self.qf1(obs, actions)
            q2_pred = self.qf2(obs, actions)
            # Make sure policy accounts for squashing functions like tanh correctly!
            new_next_actions, _, _, new_log_pi, *_ = self.policy(
                next_obs, reparameterize=True, return_log_prob=True,
            )
            target_q_values = torch.min(
                self.target_qf1(next_obs, new_next_actions),
                self.target_qf2(next_obs, new_next_actions),
            ) - alpha * new_log_pi
            
            with torch.no_grad():
                #calculate weights
                state_action = torch.cat([next_obs, new_next_actions], dim = -1)
                weight = calc_anomaly_score(self.ad, state_action, c = self.c, ad_type = self.args.ad_module)
                weight = self.weight_function(weight)

            if self.theoretically == 1:
                q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values
                qf1_loss = ((q1_pred - q_target.detach()*weight.detach())).pow(2).mean()
                qf2_loss = ((q2_pred - q_target.detach()*weight.detach())).pow(2).mean()
            elif self.theoretically == 2:
                q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values * weight.detach()
                qf1_loss = (q1_pred - q_target.detach()).pow(2).mean()
                qf2_loss = (q2_pred - q_target.detach()).pow(2).mean()
            elif self.theoretically == 3:
                q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values - self.args.weight_alpha*torch.nn.functional.softplus(weight.detach())
                qf1_loss = (q1_pred - q_target.detach()).pow(2).mean()
                qf2_loss = (q2_pred - q_target.detach()).pow(2).mean()
            else:
                q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values
                qf1_loss = ((q1_pred - q_target.detach())*weight.detach()).pow(2).mean()
                qf2_loss = ((q2_pred - q_target.detach())*weight.detach()).pow(2).mean()

            """
            Update networks
            """
            self._num_q_update_steps += 1

            self.qf1_optimizer.zero_grad()
            qf1_loss.backward(retain_graph = True)
            self.qf1_optimizer.step()

            self.qf2_optimizer.zero_grad()
            qf2_loss.backward(retain_graph = True)
            self.qf2_optimizer.step()

            """
            Soft Updates
            """
            if self._n_train_steps_total % self.target_update_period == 0:
                ptu.soft_update_from_to(
                    self.qf1, self.target_qf1, self.soft_target_tau
                )
                ptu.soft_update_from_to(
                    self.qf2, self.target_qf2, self.soft_target_tau
                )

            """
            Save some statistics for eval
            """
            if self._need_to_update_eval_statistics:
                self._need_to_update_eval_statistics = False
                """
                Eval should set this to None.
                This way, these statistics are only computed for one batch.
                """
                policy_loss = (log_pi - q_new_actions).mean()

                self.eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
                self.eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
                self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                    policy_loss
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'Q1 Predictions',
                    ptu.get_numpy(q1_pred),
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'Q2 Predictions',
                    ptu.get_numpy(q2_pred),
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'Q Targets',
                    ptu.get_numpy(q_target),
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'Log Pis',
                    ptu.get_numpy(log_pi),
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'Policy mu',
                    ptu.get_numpy(policy_mean),
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'Policy log std',
                    ptu.get_numpy(policy_log_std),
                ))
                if self.use_automatic_entropy_tuning:
                    self.eval_statistics['Alpha'] = alpha.item()
                    self.eval_statistics['Alpha Loss'] = alpha_loss.item()
            self._n_train_steps_total += 1

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [
            self.policy,
            self.qf1,
            self.qf2,
            self.target_qf1,
            self.target_qf2,
        ]

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            qf1=self.qf1,
            qf2=self.qf2,
            target_qf1=self.qf1,
            target_qf2=self.qf2,
        )