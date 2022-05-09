import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import pylab
import os
import rushhour



ngpu = 1
latent_size = 64
hidden_size = 256
image_size = 36
num_epochs = 300
batch_size = 32

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

data_loader = rushhour.dataset(bs = batch_size)






# Discriminator
D = nn.Sequential(
    nn.Linear(image_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, 1),
    nn.Sigmoid())

# Generator 
G = nn.Sequential(
    nn.Linear(latent_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, image_size),
    nn.Tanh())

G = G.to(device)
D = D.to(device)


criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.00002)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.00002)


def reset_grad():
    d_optimizer.zero_grad()
    g_optimizer.zero_grad()


d_losses = np.zeros(num_epochs)
g_losses = np.zeros(num_epochs)
real_scores = np.zeros(num_epochs)
fake_scores = np.zeros(num_epochs)



# Start training
total_step = len(data_loader)
for epoch in range(num_epochs):
    for i, images in enumerate(data_loader):
        #images = images.view(batch_size, -1).cuda()
        images = Variable(images.cuda())
        # Create the labels which are later used as input for the BCE loss
        real_labels = torch.ones(images.shape[0], 1).cuda()
        real_labels = Variable(real_labels)
        fake_labels = torch.zeros(images.shape[0], 1).cuda()
        fake_labels = Variable(fake_labels)

        # ================================================================== #
        #                      Train the discriminator                       #
        # ================================================================== #

        # Compute BCE_Loss using real images where BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))
        # Second term of the loss is always zero since real_labels == 1
        outputs = D(images.float())
        d_loss_real = criterion(outputs, real_labels)
        real_score = outputs
        
        # Compute BCELoss using fake images
        # First term of the loss is always zero since fake_labels == 0
        z = torch.randn(images.shape[0], latent_size).cuda()
        z = Variable(z)
        fake_images = G(z)
        outputs = D(fake_images)
        d_loss_fake = criterion(outputs, fake_labels)
        fake_score = outputs
        
        # Backprop and optimize
        # If D is trained so well, then don't update
        d_loss = d_loss_real + d_loss_fake
        reset_grad()
        d_loss.backward()
        d_optimizer.step()
        # ================================================================== #
        #                        Train the generator                         #
        # ================================================================== #

        # Compute loss with fake images
        z = torch.randn(images.shape[0], latent_size).cuda()
        z = Variable(z)
        fake_images = G(z)
        outputs = D(fake_images)
        
        # We train G to maximize log(D(G(z)) instead of minimizing log(1-D(G(z)))
        # For the reason, see the last paragraph of section 3. https://arxiv.org/pdf/1406.2661.pdf
        g_loss = criterion(outputs, real_labels)
        
        # Backprop and optimize
        # if G is trained so well, then don't update
        reset_grad()
        g_loss.backward()
        g_optimizer.step()
        # =================================================================== #
        #                          Update Statistics                          #
        # =================================================================== #
        d_losses[epoch] = d_losses[epoch]*(i/(i+1.)) + d_loss.item()*(1./(i+1.))
        g_losses[epoch] = g_losses[epoch]*(i/(i+1.)) + g_loss.item()*(1./(i+1.))
        real_scores[epoch] = real_scores[epoch]*(i/(i+1.)) + real_score.mean().item()*(1./(i+1.))
        fake_scores[epoch] = fake_scores[epoch]*(i/(i+1.)) + fake_score.mean().item()*(1./(i+1.))

    plt.figure()
    pylab.xlim(0, num_epochs + 1)
    plt.plot(range(1, num_epochs + 1), d_losses, label='d loss')
    plt.plot(range(1, num_epochs + 1), g_losses, label='g loss')    
    plt.legend()
    plt.savefig('loss.pdf')
    plt.close()

    plt.figure()
    pylab.xlim(0, num_epochs + 1)
    pylab.ylim(0, 1)
    plt.plot(range(1, num_epochs + 1), fake_scores, label='fake score')
    plt.plot(range(1, num_epochs + 1), real_scores, label='real score')    
    plt.legend()
    plt.savefig('accuracy.pdf')
    plt.close()
    torch.save(G.state_dict(), 'G.ckpt')
    torch.save(D.state_dict(), 'D.ckpt')