import os
import torch
from torch.nn import functional as F
import torch.nn as nn
import torchvision.transforms as transforms
import copy
import tqdm
import matplotlib.pyplot as plt

class VAE_NOVELTY(nn.Module) : 
    def __init__(self) :
        super(VAE_NOVELTY,self).__init__()
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
        self.best_model = VAE_NOVELTY()
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

    
    def fit(self, model, train_loader, valid_loader, random_state, epochs) : 
        self.lr = 1e-3
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        
        lowest_loss = 10000

        outer = tqdm.tqdm(total=epochs, desc='Epoch', position=0)

        for epoch in range(epochs) :  
            # train 
            loss_per_epoch = 0
            model.train()

            for x in train_loader :
                x = x.to(self.device)
                x_recon, mu, log_var = model(x)

                recon_loss = F.mse_loss(x_recon, x, reduction='sum')
                kld = -0.5 * torch.sum(1+log_var-mu.pow(2)-log_var.exp())
                loss = recon_loss + kld
                loss_per_epoch += loss

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
        plt.yscale('log')
        plt.legend()
        plt.savefig('training_history/novelty/randomstate_%s_epochs_%s_history' % (random_state, epochs))
        #plt.show()   
        torch.save(self.best_model.state_dict(), 'models/best_models/novelty/model_randomstate_%s_epochs_%s.pkl' % (random_state, epochs))
        
        return
        
