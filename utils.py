import numpy as np
import string
import torch
import os 

datadir = "rush.txt"


def decoder(rush):
    '''
    Takes in a rush hour string as defined in https://www.michaelfogleman.com/rush/ and transforms it into
    a 6x6 numpy array where 0 is empty, 1 is the main vehicle, -1 is a wall, and all other numbers represent the other vehicles
    '''

    board = np.zeros((6,6))

    for count, value in enumerate(rush):
        i,j = count//6 ,count%6

        if value == 'o':
            board[i][j] = 0
        elif value == 'x':
            board[i][j] = -1
        else:
            board[i][j] = string.ascii_uppercase.index(value) + 1

    return board 


def encoder(rush):

    board = rush.flatten()
    out = ""
    basestring = "xo" + string.ascii_uppercase
    for i in board:
        out += basestring[int(i+1)] 
    
    return out


def dataset(bs, short = 50000, flatten = True, new = False):
    '''
    will return the first (or random, depending on order) elements of the rush hour dataset as flattened np arrays.
    '''

    
    if short:
        if os.path.exists('rushnumpyshort.txt') and not new:
            rush = np.loadtxt('rushnumpyshort.txt')
        else:
            data = np.random.choice(np.genfromtxt(datadir, dtype= str)[:,1],short)
            rush = np.array([decoder(x).flatten() for x in data])
            np.savetxt("rushnumpyshort.txt", rush, fmt='%i')
    else: 
        if os.path.exists('rushnumpy.txt'):
            rush = np.loadtxt('rushnumpy.txt')
        else:
            data = np.genfromtxt(datadir, dtype= str)[:,1]
            rush = np.array([decoder(x).flatten() for x in data])
            np.savetxt("rushnumpy.txt", rush, fmt='%i')
    if not flatten:
        rush = rush.reshape(-1,6,6)
    data_loader = torch.utils.data.DataLoader(dataset=rush,
                                          batch_size=bs, 
                                          shuffle=True)


def dataset_wl(bs, short = 50000, flatten = True, new = False):

    '''
    will return the first (or random, depending on order) elements of the rush hour dataset as flattened np arrays.
    '''

    if short:
        if os.path.exists('rushnumpyshortwl.txt') and not new:
            rush = np.loadtxt('rushnumpyshortwl.txt')
            label =  np.loadtxt('labelnumpyshortwl.txt')
        else:
            base = np.genfromtxt(datadir, dtype= str)[:,0:2]
            data = base[np.random.choice(len(base),short)]
            rush = np.array([decoder(x[1]).flatten() for x in data])
            label = data[:,0].astype(np.int)
            np.savetxt("rushnumpyshortwl.txt", rush, fmt='%i')
            np.savetxt("labelnumpyshortwl.txt", label, fmt='%i')
    else: 
        if os.path.exists('rushnumpywl.txt'):
            rush = np.loadtxt('rushnumpywl.txt')
            label =  np.loadtxt('labelnumpywl.txt')
        else:
            data = np.genfromtxt(datadir, dtype= str)[:,0:2]
            rush = np.array([decoder(x[1]).flatten() for x in data])
            label = data[:,0].astype(np.int)
            np.savetxt("rushnumpywl.txt", rush, fmt='%i')
            np.savetxt("labelnumpywl.txt", label, fmt='%i')
    if not flatten:
        rush = rush.reshape(-1,6,6)
    
    dataset = torch.utils.data.TensorDataset(torch.tensor(rush), torch.tensor(label))
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                          batch_size=bs, 
                                          shuffle=True)

    return(data_loader)









def test_encoding():
    data = np.random.choice(np.genfromtxt(datadir, dtype= str)[:,1],50000)
    rush = np.array([decoder(x).flatten() for x in data])
    other = np.array([encoder(x) for x in rush])

    indexes = other == data

    print(np.sum(indexes))
    print(indexes.shape)

