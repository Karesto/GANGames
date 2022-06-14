import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim


from random import randint

from utils import *


bs = 32

snowdataset = dataset(32, flatten = False)
sample = next(iter(snowdataset))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
shape = sample.shape
size = shape[0]
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size = 25):
        return input.view(input.size(0), 1, size, size)


class VAE(nn.Module):
    def __init__(self, image_channels=9, h_dim=25, z_dim=32):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Conv2d(2, 1, kernel_size=3, padding=1),
            nn.ReLU(),
            Flatten()
        )
        
        self.mu = nn.Linear(h_dim*h_dim, z_dim)
        self.logvar = nn.Linear(h_dim*h_dim, z_dim)
        self.back = nn.Linear(z_dim, h_dim*h_dim)
        
        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(1, 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(2, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(4, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, image_channels, kernel_size=3, padding=1),
            nn.Softmax(),
        )
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def bottleneck(self, h):
        mu, logvar = self.mu(h), self.logvar(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.back(z)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar



cvae = VAE().to(device)
optimizer = optim.Adam(cvae.parameters(), lr=0.00005)

def loss_function(recon_x, x, mu, log_var):
    #print(recon_x,x)
    BCE = F.binary_cross_entropy(recon_x, x.float(), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD


def train(epoch):
    cvae.train()
    train_loss = 0
    for batch_idx, data in enumerate(snowdataset):
        data= data.to(device)
        optimizer.zero_grad()
        
        recon_batch, mu, log_var = cvae(data.float())
        loss = loss_function(recon_batch, data, mu, log_var)
        
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(snowdataset.dataset)))        


for epoch in range(1,1001):
    train(epoch)
    
torch.save(cvae, 'vae-recep.pth')
