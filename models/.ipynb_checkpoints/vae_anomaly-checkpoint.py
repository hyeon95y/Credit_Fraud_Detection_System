import os
import torch
from torch.nn import functional as F
import torch.nn as nn
import torchvision.transforms as transforms
import copy
import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

class TrainDataset(Dataset):

    def __init__(self, x_train, y_train, transform=None):
        self.x_train = x_train
        self.y_train = y_train

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]
    
class ValidDataset(Dataset):

    def __init__(self, x_valid, transform=None):
        self.x_valid = x_valid

    def __len__(self):
        return len(self.x_valid)

    def __getitem__(self, idx):
        return self.x_valid[idx]

class VAE_ANOMALY(nn.Module) : 
    def __init__(self) :
        super(VAE_ANOMALY,self).__init__()
        self.fc1 = nn.Linear(30, 10)
        self.fc21 = nn.Linear(10, 2)
        self.fc22 = nn.Linear(10, 2)
        self.batchnorm1 = nn.BatchNorm1d(10)
        
        self.fc3 = nn.Linear(2, 10)
        self.batchnorm3 = nn.BatchNorm1d(10)
        self.fc4 = nn.Linear(10, 30)
        self.relu = nn.ReLU()

    def encode(self, x) :
        h1 = self.batchnorm1(self.relu(self.fc1(x)))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, log_var) :
        std = torch.exp(log_var/2)
        eps = torch.rand_like(std)
        return mu+eps*std
    
    def decode(self, z) :
        h2 = self.batchnorm3(self.relu(self.fc3(z)))
        return self.fc4(h2)
    
    def forward(self, x) :
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var
    
class Trainer() : 
    
    def __init__(self, device, batch_size) :
        super(Trainer,self).__init__()
        self.best_model = VAE_ANOMALY()
        self.device = device
        self.train_loss = []
        self.valid_loss = []
        self.batch_size = batch_size
        self.best_epoch = 0
        
    def transform(self, data) : 
        data = data.to(self.device)
        self.best_model = self.best_model.to(self.device)
        recon_x, mu, log_var = self.best_model(data)
        return recon_x

    
    def fit(self, model, train_loader, valid_loader, epochs) : 
        self.lr = 1e-3
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        
        lowest_loss = 10000

        outer = tqdm.tqdm(total=epochs, desc='Epoch', position=0)

        for epoch in range(epochs) :  
            # train 
            loss_per_epoch = 0
            model.train()

            for x, y in train_loader :
                x = x.to(self.device)
                y = y.to(self.device)
                x_recon, mu, log_var = model(x)
                
                # separate normal and novelty indices
                normal_index = (y==0)
                novelty_index = (y==1)
                num_novelty = list(y[novelty_index].shape)[0]
                
                # y== 0 (normal) : minimize error
                x_normal = x[normal_index]
                x_recon_normal = x_recon[normal_index]
                mu_normal = mu[normal_index]
                log_var_normal = log_var[normal_index]
                
                recon_loss = F.mse_loss(x_recon_normal, x_normal, reduction='sum')
                kld = -0.5 * torch.sum(1 + log_var_normal - mu_normal.pow(2) - log_var_normal.exp())
                loss = recon_loss + kld
                loss_per_epoch += loss
                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                self.optimizer.step()
                
                
                # y==1 (novelty) : maximize error            
                if num_novelty != 0 :
                    #print('y : ', y)
                    #print('index of y equals 1 : ', index)
                    #print('1 only : ', y[index])
                    x_novelty = x[novelty_index]
                    x_recon_novelty = x_recon[novelty_index]
                    mu_novelty = mu[novelty_index]
                    log_var_novelty = log_var[novelty_index]
                    
                    recon_loss = F.mse_loss(x_recon_novelty, x_novelty, reduction='sum')
                    kld = -0.5 * torch.sum(1 + log_var_novelty - mu_novelty.pow(2) - log_var_novelty.exp())
                    loss = - (recon_loss + kld) # maximize error
                    #loss_per_epoch += loss     # don't count on total error
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()               
                
            self.train_loss.append(loss_per_epoch / (len(train_loader.dataset)/self.batch_size) )

            # validation
            loss_per_epoch = 0
            model.eval()
            with torch.no_grad() :
                for x in valid_loader :
                    x = x.to(self.device)
                    x_recon , mu, log_var = model(x)

                    recon_loss = F.mse_loss(x_recon, x, reduction='sum')
                    kld = -0.5 * torch.sum(1+log_var-mu.pow(2)-log_var.exp())
                    loss = recon_loss + kld
                    loss_per_epoch += recon_loss.item() + kld
            current_valid_loss = loss_per_epoch / (len(valid_loader.dataset)/self.batch_size)
            self.valid_loss.append(current_valid_loss)

            # save best model
            if epoch == 0 :
                lowest_loss = current_valid_loss
            else : 
                if current_valid_loss <= lowest_loss : 
                    lowest_loss = current_valid_loss
                    self.best_model = copy.deepcopy(model)
                    self.best_epoch = epoch

            outer.update(1)    
            
        # show training history
        plt.figure(figsize=(20, 5))
        plt.plot(self.train_loss, label='train')
        plt.plot(self.valid_loss, label='valid')
        plt.axvline(self.best_epoch, label='lowest_valid_loss', c='red')
        plt.legend()
        plt.show()
        
        return
        
