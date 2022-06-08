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
    def forward(self, input, size=1024):
        return input.view(input.size(0), size, 1, 1)

class VAE(nn.Module):
    def __init__(self, image_channels=9, h_dim=size, z_dim=4):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        self.mu = nn.Conv2d(256, z_dim, 3)
        self.logvar = nn.Conv2d(256, z_dim, 3)
        self.deassamble = nn.ConvTranspose2d(z_dim, h_dim, 3)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(h_dim, 128, kernel_size=3),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=3),
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
        z = self.deassamble(z)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar



cvae = VAE().to(device)
optimizer = optim.Adam(cvae.parameters())

def loss_function(recon_x, x, mu, log_var):
    print(recon_x[0])
    BCE = F.binary_cross_entropy(recon_x, x.float(), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD


def train(epoch):
    cvae.train()
    train_loss = 0
    for batch_idx, data in enumerate(snowdataset):
        data= data.cuda()
        optimizer.zero_grad()
        
        recon_batch, mu, log_var = cvae(data.float())
        loss = loss_function(recon_batch, data, mu, log_var)
        
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(snowdataset.dataset),
                100. * batch_idx / len(snowdataset), loss.item() / len(data)))
        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(snowdataset.dataset)))        


for epoch in range(1,301):
    train(epoch)
    
torch.save(cvae, 'vae.pth')
