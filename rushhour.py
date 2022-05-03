import torch
import numpy as np
import string
from PIL import Image, ImageDraw
import os

datadir = "rush.txt"

def transform(rush):
    '''
    Takes in a rush hour string as defined in https://www.michaelfogleman.com/rush/ and transforms it into
    a 6x6 numpy array where 0 is empty, 1 is the main vehicle, -1 is a wall, and all other numbers represent the other vehicles
    '''

    board = np.zeros((6,6))

    for count, value in enumerate(rush):
        i,j = count//6 ,count%6

        if value == 'o':
            board[i][j] = 0
        elif value == 'A':
            board[i][j] = 1
        elif value == 'x':
            board[i][j] = -1
        else:
            board[i][j] = string.ascii_uppercase.index(value)

    return board 


def dataset(bs, short = 50000, flatten = True, new = False):
    '''
    will return the first (or random, depending on order) elements of the rush hour dataset as flattened np arrays.
    '''

    
    if short:
        if os.path.exists('rushnumpyshort.txt') and not new:
            rush = np.loadtxt('rushnumpyshort.txt')
        else:
            data = np.random.choice(np.genfromtxt(datadir, dtype= str)[:,1],short)
            rush = np.array([transform(x) for x in data])
            np.savetxt("rushnumpyshort.txt", rush, fmt='%i')
    else: 
        if os.path.exists('rushnumpy.txt'):
            rush = np.loadtxt('rushnumpy.txt')
        else:
            data = np.genfromtxt(datadir, dtype= str)[:,1]
            rush = np.array([transform(x) for x in data])
            np.savetxt("rushnumpy.txt", rush, fmt='%i')
    if flatten:
        rush = rush.reshape(-1,36)
    data_loader = torch.utils.data.DataLoader(dataset=rush,
                                          batch_size=bs, 
                                          shuffle=True)

    return(data_loader)
def draw(rush):
    '''
    TODO: Finish this someday
    Takes in a rush hour board and shows an image of it
    ''' 
    height = 600
    width = 600

    boardColor = "F2EACD"
    blockedColor = "D96D60"
    gridLineColor     = "222222"
    primaryPieceColor = "CC3333"
    pieceColor        = "338899"
    pieceOutlineColor = "222222"
    labelColor        = "222222"
    wallColor         = "222222"

    step_count = 6


    image = Image.new("RGB", size=(height, width), color = boardColor) 
    draw = ImageDraw.Draw(image)

    y_start = 0
    y_end = image.height
    step_size = int(image.width / step_count)
    for x in range(0, image.width, step_size):
        line = ((x, y_start), (x, y_end))
        draw.line(line, fill=128)
    x_start = 0
    x_end = image.width

    for y in range(0, image.height, step_size):
        line = ((x_start, y), (x_end, y))
        draw.line(line, fill=128)
    del draw


