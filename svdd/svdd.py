import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from svdd.svdd_network import *
import os
import logging
    
class TrainerDeepSVDD:
    def __init__(self, args, state_dim, action_dim, data, device):
        self.args = args
        self.train_loader = data
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
    
    def pretrain(self):

        log_dir = './log/svdd/'

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        ############################################수정############################################
        logger = logging.getLogger(name=self.args.env)
        logger.setLevel(logging.INFO) ## 경고 수준 설정
        formatter = logging.Formatter('|%(asctime)s||%(name)s||%(levelname)s|\n%(message)s',
                                    datefmt='%Y-%m-%d %H:%M:%S'
                                    )
        file_handler = logging.FileHandler(log_dir+self.args.env + '_ae_pretrain' + '.log') ## 파일 핸들러 생성
        file_handler.setFormatter(formatter) ## 텍스트 포맷 설정
        logger.addHandler(file_handler) ## 핸들러 등록
        ############################################수정############################################
        
        
        
        
        # Deep SVDD에 적용할 가중치 W를 학습하기 위해 autoencoder를 학습함
        ae = C_AutoEncoder(self.state_dim, self.action_dim, self.args.hidden_dim, self.args.latent_dim).to(self.device)
        ae.apply(weights_init_normal)
        optimizer = optim.Adam(ae.parameters(), lr=self.args.lr_ad,
                               weight_decay=self.args.weight_decay_ae)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
                    milestones=self.args.lr_milestones, gamma=0.1)
        
        ae.train()
        for epoch in range(self.args.epochs_ad):
            total_loss = 0
            for x in (self.train_loader):
                x = x.float().to(self.device)
                
                optimizer.zero_grad()
                x_hat = ae(x)
                reconst_loss = torch.mean(torch.sum((x_hat - x) ** 2, dim=tuple(range(1, x_hat.dim()))))
                reconst_loss.backward()
                optimizer.step()
                
                total_loss += reconst_loss.item()
            scheduler.step()
            if epoch%10 ==0:
                print('Pretraining Autoencoder... Epoch: {}, Loss: {}'.format(
                    epoch, total_loss/len(self.train_loader)))
                logger.info(f'Pretraining Autoencoder... Epoch: {epoch}, Loss: {total_loss/len(self.train_loader)}')
        self.save_weights_for_DeepSVDD(ae, self.train_loader) 
    

    def save_weights_for_DeepSVDD(self, model, dataloader):
        
        # AE의 encoder 구조의 가중치를 Deep SVDD에 초기화하기 위함임
        c = self.set_c(model, dataloader)
        net = network(self.state_dim, self.action_dim, self.args.hidden_dim, self.args.latent_dim).to(self.device)
        state_dict = model.state_dict()
        # 구조가 맞는 부분만 가중치를 load함
        net.load_state_dict(state_dict, strict=False)
        torch.save({'center': c.cpu().data.numpy().tolist(),
                    'net_dict': net.state_dict()}, os.path.join(self.args.ad_save_path, 'svdd', self.args.env, 'pretrained_parameters_' + self.args.env +'.pth'))
    

    def set_c(self, model, dataloader, eps=0.1):
        
        # 구의 중심점을 초기화함
        model.eval()
        z_ = []
        with torch.no_grad():
            for x in dataloader:
                x = x.float().to(self.device)
                z = model.encoder(x)
                z_.append(z.detach())
        z_ = torch.cat(z_)
        c = torch.mean(z_, dim=0)
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        return c


    def train(self):
        
        # AE의 학습을 마치고 그 가중치를 적용한 Deep SVDD를 학습함
        net = network(self.state_dim, self.action_dim, self.args.hidden_dim, self.args.latent_dim).to(self.device)
        
        if self.args.ad_train==True:
            state_dict = torch.load(os.path.join(self.args.ad_save_path, 'svdd', self.args.env, 'pretrained_parameters_' + self.args.env +'.pth'))
            net.load_state_dict(state_dict['net_dict'])
            c = torch.Tensor(state_dict['center']).to(self.device)
        else:
            # pretrain을 하지 않았을 경우 가중치를 초기화함
            net.apply(weights_init_normal)
            c = torch.randn(self.args.latent_dim).to(self.device)
        
        optimizer = optim.Adam(net.parameters(), lr=self.args.lr_ad,
                            weight_decay=self.args.weight_decay_svdd)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
                    milestones=self.args.lr_milestones, gamma=0.1)

        net.train()
        for epoch in range(self.args.epochs_ad):
            total_loss = 0
            for x in (self.train_loader):
                x = x.float().to(self.device)

                optimizer.zero_grad()
                z = net(x)
                loss = torch.mean(torch.sum((z - c) ** 2, dim=1))
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            scheduler.step()

            if epoch%10==0:
                print('Training Deep SVDD... Epoch: {}, Loss: {}'.format(
                    epoch, total_loss/len(self.train_loader)))
        torch.save(net.state_dict(), os.path.join(self.args.ad_save_path, 'svdd', self.args.env, 'SVDD_' + self.args.env +'.pth'))
        self.net = net
        self.c = c
    
    def bound(self):
        
        net = network(self.state_dim, self.action_dim, self.args.hidden_dim, self.args.latent_dim).to(self.device)
        
        state_dict = torch.load(os.path.join(self.args.ad_save_path, 'svdd',  self.args.env, 'pretrained_parameters_' + self.args.env +'.pth'))
        c = torch.Tensor(state_dict['center']).to(self.device)
        
        net.load_state_dict(torch.load(os.path.join(self.args.ad_save_path, 'svdd', self.args.env, 'SVDD_' + self.args.env +'.pth')))
        
        net.eval()
        i = 0
        for x in (self.train_loader):
            x = x.float().to(self.device)
            z = net(x)
            z = torch.sum((z-c)**2, dim=1)
            z = z.cpu().tolist()
            if i ==0:
                result = z
            else:
                result = np.concatenate((result, z))
            i+=1
            
        result = result.reshape(-1)
        return np.max(result)
    
    def load_SVDD(self):
        svdd = network(self.state_dim, self.action_dim, self.args.hidden_dim, self.args.latent_dim)

        state_dict = torch.load(os.path.join(self.args.ad_save_path , 'svdd', self.args.env, 'pretrained_parameters_' + self.args.env +'.pth'))
        c = torch.Tensor(state_dict['center'])
        svdd.load_state_dict(torch.load(os.path.join(self.args.ad_save_path, 'svdd', self.args.env, 'SVDD_' + self.args.env +'.pth')))

        return svdd, c