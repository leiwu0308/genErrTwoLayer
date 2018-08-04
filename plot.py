import os
from collections import defaultdict
import numpy as np 
import matplotlib.pyplot as plt


from models import TwoLayerNet
from trainer import eval
from data import load_cifar, load_mnist


BATCH_SIZE = 100
NSAMPLES = 100

# 使用数据增强
train_dl, test_dl = load_cifar(BATCH_SIZE,
                    nsamples=NSAMPLES,
                    root='./data/cifar10/',
                    nclasses=2,one_hot=2)


# load data
filenames_raw = os.listdir('./checkpoints/mnist_100/')
res  = defaultdict(list)

i = 0
for filename in filenames_raw:
    if filename.startswith('cifar'):
        continue
    dd = filename.split('_')
    width = int(dd[1][5:])
    lmbd = float(dd[2][4:])
    net = TwoLayerNet(784,width,10)
    net.load_state_dict(torch.load('checkpoints/mnist_100/'+filename))
    net = net.cuda()
    
    pthnorm = net.path_norm().data[0]
    tr_loss, tr_acc = eval(net,criterion,train_dl)
    te_loss, te_acc = eval(net,criterion,test_dl)
    
    res[lmbd].append((width,pthnorm,tr_acc,te_acc,tr_loss,te_loss))
    
   





