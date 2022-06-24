"""
In ResNet, we see how the skip connection added as identity function from the inputs
to interact with the Conv layers. But in DenseNet, we see instead of adding skip 
connection to Conv layers, we can append or concat the output of identity function
with output of Conv layers.
In ResNet, it is little tedious to make the dimensions to match for adding the skip
connection and Conv Layers, but it is much simpler in DenseNet, as we concat the 
both the X and Conv's output.
The key idea or the reason its called DenseNet is because the next layers not only get
the input from previous layer but also preceeding layers before the previous layer. So 
the next layer becomes dense as it loaded with output from previous layers.
Check Figure 7.7.2 from https://d2l.ai/chapter_convolutional-modern/densenet.html for 
why DenseNet is Dense?
Two blocks comprise DenseNet, one is DenseBlock for concat operation and other is 
transition layer for controlling channels meaning dimensions (recall 1x1 Conv).
"""
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
"""
In ResNet, we see how the skip connection added as identity function from the inputs
to interact with the Conv layers. But in DenseNet, we see instead of adding skip 
connection to Conv layers, we can append or concat the output of identity function
with output of Conv layers.
In ResNet, it is little tedious to make the dimensions to match for adding the skip
connection and Conv Layers, but it is much simpler in DenseNet, as we concat the 
both the X and Conv's output.
The key idea or the reason its called DenseNet is because the next layers not only get
the input from previous layer but also preceeding layers before the previous layer. So 
the next layer becomes dense as it loaded with output from previous layers.
Check Figure 7.7.2 from https://d2l.ai/chapter_convolutional-modern/densenet.html for 
why DenseNet is Dense?
Two blocks comprise DenseNet, one is DenseBlock for concat operation and other is 
transition layer for controlling channels meaning dimensions (recall 1x1 Conv).
"""
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import argparse
from utils import *

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")




class Bottleneck(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(Bottleneck, self).__init__()
        interChannels = 4*growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1,
                               padding='same', bias=False)
        self.bn2 = nn.BatchNorm2d(interChannels)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat((x, out), 1)
        return out

class SingleLayer(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out

class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1,
                               padding='same',bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        #out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(self, input_channel, growthRate, depth, reduction, n_classes, bottleneck):
        super(DenseNet, self).__init__()

        nDenseBlocks = (depth-4) // 3
        if bottleneck:
            nDenseBlocks //= 2
        nChannels = 2*growthRate
        self.conv1 = nn.Conv2d(input_channel, nChannels, kernel_size=3, padding=1,
                               bias=False)
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans1 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans2 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans3 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense4 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nOutChannels += nDenseBlocks*growthRate

        nChannels = nOutChannels
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.fc = nn.Linear(nChannels, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate))
            else:
                layers.append(SingleLayer(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.relu(self.bn1(out))
        out = torch.squeeze(F.adaptive_avg_pool2d(out, 1))
        out = F.softmax(self.fc(out))

        return out






def train_model(net, train_loader, pth_filename, num_epochs, val_loader = None):
    print("Starting training")
    learning_rate = 0.0005

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)  

    for epoch in range(num_epochs):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            targets = torch.nn.functional.one_hot(targets.to(torch.int64), num_classes = 60)
            optimizer.zero_grad()
            outputs = net(inputs.float())
            loss = criterion(outputs, targets.float())
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets.argmax(1)).sum().item()
            if batch_idx % 500 ==0 : 
                print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        print(predicted, targets.argmax(1))
        scheduler.step()


    torch.save(net.state_dict(),pth_filename)
    print('Model saved in {}'.format(pth_filename))

def main():
    input_channel = 26
    n_classes = 60

    datadir = "rush.txt"
    ngpu = 1


    batch_size = 32
    
    #### Create model and move it to whatever device is available (gpu/cpu)
    model = DenseNet(input_channel=input_channel, n_classes=n_classes, 
            growthRate=12, depth=40, reduction=0.5, bottleneck=True).to(device)
    model.to(device)

    train_loader = dataset_wl(batch_size, short = 5000)

    #### Model training (if necessary)
    train_model(model, train_loader, "densenet1.pth", 100)



if __name__ == "__main__":
    main()
