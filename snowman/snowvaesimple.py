import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import utils


bs = 32

snowdataset = utils.dataset(32, flatten = True, new = True)
sample = next(iter(snowdataset))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
shape = sample.shape
size = shape[1]


class CVAE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super(CVAE, self).__init__()
        self.x_dim = x_dim
        # encoder part
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)
        self.fc32 = nn.Linear(h_dim2, z_dim)
        # decoder part
        self.fc4 = nn.Linear(z_dim, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)
    
    def encoder(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h)
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add(mu) # return z sample
    
    def decoder(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        recon = F.softmax(self.fc6(h).view(-1,9,25,25),dim=1)

        return recon.view(-1,5625)
    
    def forward(self, x):
        mu, log_var = self.encoder(x.view(-1, self.x_dim))
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var


cvae = CVAE(x_dim=size, h_dim1=128, h_dim2=64, z_dim=16)
if torch.cuda.is_available():
    cvae.cuda()



optimizer = optim.Adam(cvae.parameters())
# return reconstruction error + KL divergence losses

def loss_function(recon_x, x, mu, log_var):
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
        
        # if batch_idx % 100 == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(snowdataset.dataset),
        #         100. * batch_idx / len(snowdataset), loss.item() / len(data)))
        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(snowdataset.dataset)))        


for epoch in range(1,301):
    train(epoch)
    
torch.save(cvae, 'vae.pth')
