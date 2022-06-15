import numpy as np
import string
import torch
import os 


path = "levels_walled.txt"

def decoder(snow):
    '''
    Decodes a snow string matrix to a multi channel matrix
    '''
    size = snow.shape[0]
    outs = (9,size,size)
    board = np.zeros(outs)

    for i in range(size):
        for j in range(size):

            if snow[i][j] == 's':
                board[0][i][j] = 1
            elif snow[i][j] == 'b':
                board[1][i][j] = 1
            elif snow[i][j] == 'c':
                board[2][i][j] = 1
            elif snow[i][j] == 'h':
                board[3][i][j] = 1
            elif snow[i][j] == 'k':
                board[4][i][j] = 1
            elif snow[i][j] == 'l':
                board[5][i][j] = 1
            elif snow[i][j] == '.':
                board[6][i][j] = 1
            elif snow[i][j] == 'A':
                board[7][i][j] = 1
            elif snow[i][j] == 'W':
                board[8][i][j] = 1
            

    return board 

def encoder(snow):
    
    size = snow.shape[1]
    board = np.empty((size, size), dtype='<U1').astype("str")
    order = ['s', 'b', 'c', 'h', 'k', 'l', '.', 'A', 'W']
    for i in range(9):
        board = np.core.defchararray.add(board,np.where(snow[i], np.full((size,size),order[i]), ''))

    return board

def dataset(bs, flatten = True, new = True):
    '''
    will return the first (or random, depending on order) elements of the rush hour dataset as flattened np arrays.
    '''

    
    if os.path.exists('snownumpy.txt') and not new:
        snow = np.loadtxt('snownumpy.txt')
        size = snow.shape[1]
    else:
        with open(path, 'r', newline='\n') as f:
            data = np.genfromtxt(f, dtype= str)
        datacomp = np.array(list(map(lambda x: list(x),data)))
        size = datacomp.shape[1]
        datacomp = datacomp.reshape(-1, size, size)
        snow = np.array([decoder(x).flatten() for x in datacomp])
        np.savetxt("snownumpy.txt", snow, fmt='%i')
    if not flatten:
        snow = snow.reshape(-1,9, size,size)
    data_loader = torch.utils.data.DataLoader(dataset=snow,
                                          batch_size=bs, 
                                          shuffle=True)
    return data_loader



def test_encoding():
    with open(path, 'r', newline='\n') as f:
        data = np.genfromtxt(f, dtype= str)
    datacomp = np.array(list(map(lambda x: list(x),data)))
    size = datacomp.shape[1]
    datacomp = datacomp.reshape(-1, size, size)
    snow = np.array([decoder(x) for x in datacomp])
    other = np.array([encoder(x) for x in snow])
    print(other.shape, datacomp.shape)
    indexes = other == datacomp

    print(np.sum(indexes))
    print(indexes.shape)

def frange_cycle_sigmoid(start, stop, n_epoch, n_cycle=4, ratio=0.5):
    L = np.ones(n_epoch)
    period = n_epoch/n_cycle
    step = (stop-start)/(period*ratio) # step is in [0,1]
    
    # transform into [-6, 6] for plots: v*12.-6.

    for c in range(n_cycle):

        v , i = start , 0
        while v <= stop:
            L[int(i+c*period)] = 1.0/(1.0+ np.exp(- (v*12.-6.)))
            v += step
            i += 1
    return L    

def frange_cycle_cosine(start, stop, n_epoch, n_cycle=4, ratio=0.5):
    L = np.ones(n_epoch)
    period = n_epoch/n_cycle
    step = (stop-start)/(period*ratio) # step is in [0,1]
    
    # transform into [0, pi] for plots: 

    for c in range(n_cycle):

        v , i = start , 0
        while v <= stop:
            L[int(i+c*period)] = 0.5-.5*np.cos(v*np.pi)
            v += step
            i += 1
    return L  