import sched
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt

from random import randint

from utils import *


bs = 16

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
    def __init__(self, image_channels=9, h_dim=25, z_dim=[512,256,256]):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 8, kernel_size=3, padding=1),
            #nn.BatchNorm2d(8),
            nn.LeakyReLU(0.1),
            nn.Conv2d(8, 4, kernel_size=3, padding=1),
            #nn.BatchNorm2d(4),
            nn.LeakyReLU(0.1),
            nn.Conv2d(4, 2, kernel_size=3, padding=1),
            #nn.BatchNorm2d(2),
            nn.LeakyReLU(0.1),
            nn.Conv2d(2, 1, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Dropout2d(p=0.3),
            Flatten(),
            nn.Linear(h_dim*h_dim, z_dim[0]),
            nn.LeakyReLU(0.1),
            nn.Linear(z_dim[0], z_dim[1]),
            nn.LeakyReLU(0.1)
        )
        
        self.mu = nn.Linear(z_dim[1], z_dim[2])
        self.logvar = nn.Linear(z_dim[1], z_dim[2])
        
        self.back = nn.Sequential(
            nn.Linear(z_dim[2], z_dim[1]),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=0.2),
            nn.Linear(z_dim[1], z_dim[0]),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=0.2),
            nn.Linear(z_dim[0],h_dim*h_dim),
            nn.LeakyReLU(0.1)
        )
        
        self.decoder = nn.Sequential(
            
            UnFlatten(),
            nn.ConvTranspose2d(1, 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(2, 4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(4, 8, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(8, image_channels, kernel_size=3, padding=1),
            nn.Softmax2d(),
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
optimizer = optim.Adam(cvae.parameters(), lr=0.0025, weight_decay= 0.01)
lmbda = lambda epoch: 0.99
scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)
loss = torch.nn.BCELoss()
#   loss = DiceLoss()

def loss_function(recon_x, x, mu, log_var, coeff = 1):

    #print(recon_x[0],x[0])
    #BCE = F.binary_cross_entropy(recon_x, x.float())
    BCE = loss(recon_x,x.float())
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + coeff*KLD


def train(epoch, coeff):
    cvae.train()
    train_loss = 0
    for batch_idx, data in enumerate(snowdataset):
        data= data.to(device)
        #data = torch.ones_like(data)*00.3569
        optimizer.zero_grad()
        
        recon_batch, mu, log_var = cvae(data.float())
        loss = loss_function(recon_batch, data, mu, log_var, coeff)
        
        loss.backward()
        train_loss += loss.item()
        optimizer.step()


    elem = recon_batch
    pred = torch.argmax(elem, dim = 1)
    out = torch.zeros_like(elem).scatter_(1, pred.unsqueeze(1), 1.).cpu().numpy()
    print(encoder(out[0]))

    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss))   
    scheduler.step()

n_epoch = 50
cycle = frange_cycle_cosine(0,1,n_epoch)

for epoch in range(1,n_epoch+1):
    train(epoch, cycle[epoch-1])
    
torch.save(cvae, 'vae-recep.pth')
