import numpy as np
import string
import torch
import os 



datadir = "data/rush.txt"


def decoder(rush, size = 6):
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

def decoder_one_hot(rush, size = 6):

    board = np.zeros((26,6,6))

    for count, value in enumerate(rush):
        i,j = count//6 ,count%6

        if value == 'o':
            board[1][i][j] = 1
        elif value == 'x':
            board[0][i][j] = 1
        else:
            index = string.ascii_uppercase.index(value)+2
            board[index][i][j] = 1

    return board 

def encoder_one_hot(rush):
    
    out = ""
    n = rush.shape[1]
    basestring = "xo" + string.ascii_uppercase

    for i in range(n):
        for j in range(n):
            index =np.where(rush[:,i,j]==1)[0][0]
            out += basestring[index]
            
    return out


def encoder(rush, first, size = 6):
    if first:
        rush = np.rot90(rush.reshape((size,size)),2).T 
        rush = rush[::-1,:]
    board = rush.flatten()
    out = ""
    basestring = "xo" + string.ascii_uppercase
    for i in board:
        out += basestring[int(i+1)] 
    print(out)
    return out


def dataset(bs, short = 50000, flatten = False, new = False):
    '''
    will return the first (or random, depending on order) elements of the rush hour dataset as flattened np arrays.
    '''

    
    if short:
        if os.path.exists('data/rushnumpyshort.txt') and not new:
            rush = np.loadtxt('data/rushnumpyshort.txt')
        else:
            data = np.random.choice(np.genfromtxt(datadir, dtype= str)[:,1],short)
            rush = np.array([decoder(x).flatten() for x in data])
            np.savetxt("data/rushnumpyshort.txt", rush, fmt='%i')
    else: 
        if os.path.exists('data/rushnumpy.txt'):
            rush = np.loadtxt('data/rushnumpy.txt')
        else:
            data = np.genfromtxt(datadir, dtype= str)[:,1]
            rush = np.array([decoder(x).flatten() for x in data])
            np.savetxt("data/rushnumpy.txt", rush, fmt='%i')
    if not flatten:
        rush = rush.reshape(-1,6,6)
    data_loader = torch.utils.data.DataLoader(dataset=rush,
                                          batch_size=bs, 
                                          shuffle=True)


def dataset_wl(bs, short = 50000, flatten = False, new = False):

    '''
    will return the first (or random, depending on order) elements of the rush hour dataset as flattened np arrays.
    '''

    if short:
        if os.path.exists('data/rushnumpyshortwl.npy') and not new:
            rush = np.load('data/rushnumpyshortwl.npy')
            label =  np.load('data/labelnumpyshortwl.npy')
        else:
            base = np.genfromtxt(datadir, dtype= str)[:,0:2]
            data = base[np.random.choice(len(base),short)]
            rush = np.array([decoder_one_hot(x[1]).flatten() for x in data])
            label = data[:,0].astype(np.int)
            np.save("data/rushnumpyshortwl.npy", rush)
            np.save("data/labelnumpyshortwl.npy", label)
    else: 
        if os.path.exists('data/rushnumpywl.npy'):
            rush = np.load('data/rushnumpywl.npy')
            label =  np.load('data/labelnumpywl.npy')
        else:
            data = np.genfromtxt(datadir, dtype= str)[:,0:2]
            rush = np.array([decoder_one_hot(x[1]).flatten() for x in data])
            label = data[:,0].astype(np.int)
            np.save("data/rushnumpywl.npy", rush)
            np.save("data/labelnumpywl.npy", label)
    if not flatten:
        rush = rush.reshape(-1,26,6,6)
    
    dataset = torch.utils.data.TensorDataset(torch.tensor(rush), torch.tensor(label))
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                          batch_size=bs, 
                                          shuffle=True)

    return(data_loader)

def datasetwithval(bs, num = 100000, flatten = False, new = False):

    if os.path.exists('data/rushtest.npy') and not new:
        rush = np.load('data/rushtest.npy')
        label =  np.load('data/labeltest.npy')
    else:
        base = np.genfromtxt(datadir, dtype= str)[:,0:2]
        data = base[np.random.choice(len(base),num)]
        rush = np.array([decoder_one_hot(x[1]).flatten() for x in data])
        label = data[:,0].astype(np.int)
        np.save("data/rushtest.npy", rush)
        np.save("data/labeltest.npy", label)

    
    if not flatten:
        rush = rush.reshape(-1,26,6,6)
    ratio = 0.2
    dataset = torch.utils.data.TensorDataset(torch.tensor(rush), torch.tensor(label))
    train_set, val_set = torch.utils.data.random_split(dataset, [int(num*(1-ratio)), int(num*ratio)], torch.Generator().manual_seed(42))
    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                          batch_size=bs, 
                                          shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_set,
                                          batch_size=bs, 
                                          shuffle=True)
    return(train_loader, val_loader)




def data_oracle(bs, num = 100000, flatten = False, new = False):

    

    if os.path.exists('data/rushtest.npy') and not new:
        rush = np.load('data/rushtest.npy')
        label =  np.load('data/labeltest.npy')
    else:
        unsolvable = np.load("data/unsolvable_lvl1.npy")
        num = min(num,unsolvable.shape[0])
        base = np.genfromtxt(datadir, dtype= str)[:,0:2]
        data = base[np.random.choice(len(base),num)]
        rush = np.array([decoder_one_hot(x[1]).flatten() for x in data]+ [decoder_one_hot(x).flatten() for x in unsolvable])
        label = np.array([1]*num + [0]*num)
        np.save("data/rushtest.npy", rush)
        np.save("data/labeltest.npy", label)

    
    if not flatten:
        rush = rush.reshape(-1,26,6,6)

    #Making sure of integer length for both sets
    ratio = 0.2
    length = rush.shape[0]
    train_length = int(length*(1-ratio))
    val_length = length - train_length

    #Dividing the set into train/test (or val)
    dataset = torch.utils.data.TensorDataset(torch.tensor(rush), torch.tensor(label))
    train_set, val_set = torch.utils.data.random_split(dataset, [train_length, val_length], torch.Generator().manual_seed(42))
    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                          batch_size=bs, 
                                          shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_set,
                                          batch_size=bs, 
                                          shuffle=True)
    return(train_loader, val_loader)



def data_oracle2(bs, num = 100000, flatten = False, new = False, cat = None):

    

    data = np.load("data/randombase.npy")
    label = np.load("data/randombaselabel.npy")*1
    rush = np.array([decoder_one_hot(x[1]).flatten() for x in data])
    if not flatten:
        rush = rush.reshape(-1,26,6,6)
    if cat:
        label = (label/60).astype(np.int) * cat
    #Making sure of integer length for both sets
    ratio = 0.2
    length = rush.shape[0]
    train_length = int(length*(1-ratio))
    val_length = length - train_length

    #Dividing the set into train/test (or val)
    dataset = torch.utils.data.TensorDataset(torch.tensor(rush), torch.tensor(label))



    train_set, val_set = torch.utils.data.random_split(dataset, [train_length, val_length], torch.Generator().manual_seed(42))
    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                            sampler=ImbalancedDatasetSampler(train_set),
                                          batch_size=bs, 
                                          shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_set,
                                          batch_size=bs, 
                                          shuffle=True)
    return(train_loader, val_loader)

def data_solver(bs, num = 5000, flatten = False, new = False, cat = None):

    

    if os.path.exists('data/rushtest.npy') and not new:
        rush = np.load('data/rushtest.npy')
        label =  np.load('data/labeltest.npy')
    else:
        base = np.genfromtxt(datadir, dtype= str)[:,0:2]
        data = base[np.random.choice(len(base),num)]
        rush = np.array([decoder_one_hot(x[1]).flatten() for x in data])
        label = data[:,0].astype(np.int)
        np.save("data/rushtest.npy", rush)
        np.save("data/labeltest.npy", label)

    
    if not flatten:
        rush = rush.reshape(-1,26,6,6)
    
    if cat:
        label = (label/np.max(label)* cat).astype(np.int) 
    #Making sure of integer length for both sets
    ratio = 0.2
    length = rush.shape[0]
    train_length = int(length*(1-ratio))
    val_length = length - train_length

    class_sample_count = np.array(
    [len(np.where(label == t)[0]) for t in np.unique(label)])
    print(np.unique(label))
    weight = 1. / class_sample_count
    print(weight)
    samples_weight = torch.from_numpy(weight)
    sampler = torch.utils.data.WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))
    #Dividing the set into train/test (or val)
    dataset = torch.utils.data.TensorDataset(torch.tensor(rush), torch.tensor(label))




    train_set, val_set = torch.utils.data.random_split(dataset, [train_length, val_length], torch.Generator().manual_seed(42))
    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                            sampler=sampler,
                                          batch_size=bs)
    val_loader = torch.utils.data.DataLoader(dataset=val_set,
                                          batch_size=bs, 
                                          shuffle=True)
    return(train_loader, val_loader)












def test_encoding():
    data = np.random.choice(np.genfromtxt(datadir, dtype= str)[:,1],50000)
    rush = np.array([decoder(x).flatten() for x in data])
    other = np.array([encoder(x) for x in rush])

    indexes = other == data

    print(np.sum(indexes))
    print(indexes.shape)

def test_onehot_encoding():
    data = np.random.choice(np.genfromtxt(datadir, dtype= str)[:,1],50000)
    rush = np.array([decoder_one_hot(x) for x in data])
    print(rush[0].shape)

    other = np.array([encoder_one_hot(x) for x in rush])

    indexes = other == data

    print(np.sum(indexes))
    print(indexes.shape)

def transform(path):
    rush = np.loadtxt(path) + 1 
    rush_enc = np.array([encoder(x) for x in rush])
    print(rush_enc)
    np.savetxt("data/unsolvabletxt.txt", rush_enc, fmt='%s')
    np.save("data/unsolvable_lvl1.npy",rush_enc)

def transform2(path):
    data = np.loadtxt(path, dtype = str)
    rush = data.astype(np.float) +1
    print(rush.shape)
    rush_enc = np.array([encoder(x, first= True,size= 8) for x in rush])
    np.savetxt("lvl8txt.txt", rush_enc, fmt='%s')
    np.save("data/lvl8.npy", rush_enc)
    #np.save("data/lvl8.npy", label)

def gaspard_to_fogleman(path):
    data = np.loadtxt(path, dtype = str)
    size = int(np.sqrt(data.shape[1]))
    rush = data.astype(np.float) +1
    
    rush_encoded = np.array([encoder(x, first= True,size= size) for x in rush])
    outpath = os.path.splitext(path)[0] + "translated"
    
    np.savetxt(outpath + ".txt", rush_encoded, fmt='%s')
    np.save(outpath + ".npy", rush_encoded)

# transform("unsolvables_lvl1.txt")
#test_onehot_encoding()

# gaspard_to_fogleman("data/solv_black_boxes_6x6.txt")
# gaspard_to_fogleman("data/unsolv_black_boxes_6x6.txt")

# transform2("data/lvl8.txt")

