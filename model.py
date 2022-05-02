import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F

init_channels = 64
image_channels = 3
latent_dim = 100

class CAE(nn.Module):
    def __init__(self) -> None:
        super(CAE, self).__init__()
        self.name = 'CAE'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.encoder = nn.Sequential(
        nn.Conv2d(in_channels=image_channels,out_channels=init_channels,kernel_size=4,stride=2,padding=2),
        nn.ReLU(),
        nn.Conv2d(in_channels=init_channels,out_channels=init_channels*2,kernel_size=4,stride=2,padding=2),
        nn.ReLU(),
        nn.Conv2d(in_channels=init_channels*2,out_channels=init_channels*4,kernel_size=4,stride=2,padding=2),
        nn.ReLU(),
        nn.Conv2d(in_channels=init_channels*4,out_channels=init_channels*8,kernel_size=4,stride=2,padding=2),
        nn.ReLU(),
        nn.Conv2d(in_channels=init_channels*8,out_channels=init_channels*16,kernel_size=4,stride=2,padding=2),
        nn.ReLU(),
        nn.Conv2d(in_channels=init_channels*16,out_channels=1024,kernel_size=4,stride=2,padding=2),
        nn.ReLU(),

        )

        self.fc1 = nn.Linear(1024, 2048)
        self.fc_mu = nn.Linear(2048, latent_dim)
        self.fc_log_var = nn.Linear(2048, latent_dim)
        self.fc2 = nn.Linear(latent_dim, 1024)

        
        self.decoder = nn.Sequential(
        nn.ConvTranspose2d(in_channels=1024,out_channels=init_channels*16,kernel_size=3,stride=2),
        nn.ReLU(),
        nn.ConvTranspose2d(in_channels=init_channels*16,out_channels=init_channels*8,kernel_size=3,stride=2),
        nn.ReLU(),
        nn.ConvTranspose2d(in_channels=init_channels*8,out_channels=init_channels*4,kernel_size=3,stride=2),
        nn.ReLU(),
        nn.ConvTranspose2d(in_channels=init_channels*4,out_channels=init_channels*2,kernel_size=3,stride=2),
        nn.ReLU(),
        nn.ConvTranspose2d(in_channels=init_channels*2,out_channels=init_channels,kernel_size=3,stride=2),
        nn.ReLU(),
        nn.ConvTranspose2d(in_channels=init_channels,out_channels=image_channels,kernel_size=3,stride=2, output_padding=1),
        nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        #print(f"After encoder: {x.shape}")

        x = F.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        #print(f"After pooling: {x.shape}")
        h = self.fc1(x)
        #print(f"Shape of h: {h.shape}")
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        #print(f"Shape of mu: {mu.shape}")
        z = self.reparam(mu, log_var)
        z = self.fc2(z)
        #print(f"Shape of z: {z.shape}")
        z = z.view(z.shape[0],1024,1,1)
        #print(out.shape)

        out = self.decoder(z)
        return out, mu, log_var

    def reparam(self, mu, log_var):
        # Reparameterization function...!
        sigma = torch.exp(0.5*log_var)
        batch, dim = sigma.shape
        eps = torch.randn((batch, dim), device=self.device)
        out = mu + (eps * sigma) 
        return out

    @torch.no_grad()
    def generate(self):
        mu = torch.randn(10, 100, device=self.device)
        log_var = torch.randn(10, 100, device=self.device)
        z = self.reparam(mu, log_var)
        z = self.fc2(z)
        #print(f"Shape of z: {z.shape}")
        z = z.view(z.shape[0],1024,1,1)
        #print(out.shape)
        out = self.decoder(z)
        return out

    @torch.no_grad()
    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.shape[0], -1)
        mu = self.mu(x)
        sigma = F.relu(self.sigma(x))
        out = self.compute_latent(mu, sigma)
        return out


class lstm_ae(nn.Module):
    def __init__(self,hyper_param, model_param)-> None:
        super(lstm_ae, self).__init__()
        self.name = 'lstm_ae'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        in_units = model_param['in_units']
        out_units = model_param['out_units']
        self.t_out = model_param['t_out']

        latent_dim = hyper_param['hidden_dim'] 
        drop = hyper_param['drop']

        self.encoder = nn.LSTM(in_units, latent_dim, 2, bidirectional=False, dropout=drop)
        self.fc1 = nn.Linear(latent_dim,latent_dim)
        # self.fc2 = nn.Linear(latent_dim, out_units)
        self.decoder1 = nn.LSTM(latent_dim, out_units, 2, batch_first=True,dropout=drop)
        # self.decoder2 =  nn.LSTM(out_units, out_units, 1, batch_first=True,dropout=drop)
        

    def forward(self, x):
        x,z = self.encoder(x)

        #pick only the last output sample
        latent = self.fc1(x[:,:,:])

        # d_in = torch.zeros((x.shape[0], self.t_out, 1), device=self.device)
        # h = self.fc2(latent).unsqueeze(dim=0)
        # h = torch.cat((h,h), dim=0)
        # out = self.decoder1(d_in, (h,h))
        # out,_ = self.decoder2(*out)
        out,_ = self.decoder1(latent)
        return torch.exp(out)
    
    def predict(self, x):
        x,z = self.encoder(x)

        #pick only the last output sample
        latent = self.fc1(x[:,:,:])

        # d_in = torch.zeros((x.shape[0], self.t_out, 1), device=self.device)
        # h = self.fc2(latent).unsqueeze(dim=0)
        # h = torch.cat((h,h), dim=0)
        # out = self.decoder1(d_in, (h,h))
        # out,_ = self.decoder2(*out)
        out,_ = self.decoder1(latent)
        return torch.exp(out)
