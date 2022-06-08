import numpy as np
import string
import torch
import os 


path = "levels.txt"

def decoder(snow):
    '''
    Decodes a snow string matrix to a multi channel matrix
    '''
    size = snow.shape[0]
    outs = (size,size,9)
    board = np.zeros(outs)

    for i in range(size):
        for j in range(size):

            if snow[i][j] == 's':
                board[i][j][0] = 1
            elif snow[i][j] == 'b':
                board[i][j][1] = 1
            elif snow[i][j] == 'c':
                board[i][j][2] = 1
            elif snow[i][j] == 'h':
                board[i][j][3] = 1
            elif snow[i][j] == 'k':
                board[i][j][4] = 1
            elif snow[i][j] == 'l':
                board[i][j][5] = 1
            elif snow[i][j] == '.':
                board[i][j][6] = 1
            elif snow[i][j] == 'A':
                board[i][j][7] = 1
            elif snow[i][j] == 'W':
                board[i][j][8] = 1
            

    return board 

def encoder(snow):
    
    size = snow.shape[0]
    board = np.empty((size, size), dtype='<U1').astype("str")
    order = ['s', 'b', 'c', 'h', 'k', 'l', '.', 'A', 'W']
    for i in range(9):
        board = np.core.defchararray.add(board,np.where(snow[:,:,i], np.full((size,size),order[i]), ''))

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

