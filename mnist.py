import time
import argparse
import math
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.autograd import Variable

from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as trans

from models import TwoLayerNet
from trainer import *

argparser = argparse.ArgumentParser()
argparser.add_argument('--batchsize',type=int,default=100)
argparser.add_argument('--width',type=int,default=1000)
argparser.add_argument('--nepochs',type=int,default=100)
argparser.add_argument('--lr',type=float,default=0.01)
argparser.add_argument('--lmbd', type=float, default=0.0)
argparser.add_argument('--nsample', type=int, default=60000)
args = argparser.parse_args()

# Data Load
train_set = dsets.MNIST(root='./data/mnist',train=True,
                          transform=trans.Compose([
                              trans.ToTensor()
                          ]))
train_dl = DataLoader(train_set,batch_size=args.batchsize,shuffle=True,num_workers=6)
train_set.train_data = train_set.train_data[0:args.nsample]
train_set.train_labels = train_set.train_labels[0:args.nsample]

test_set = dsets.MNIST(root='./data/mnist',train=False,
                         transform=trans.Compose([
                             trans.ToTensor()
                         ]))
test_dl = DataLoader(test_set,batch_size=args.batchsize,num_workers=6)


# Train the model
m = args.width
nepochs = args.nepochs
lr = args.lr

net = TwoLayerNet(784,m,10)
net = net.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(),
                            lr=lr,momentum=0.9,nesterov=True)
scheduler = lr_scheduler.MultiStepLR(optimizer,
                          milestones=[int(nepochs*0.7),int(nepochs*0.9)],
                          gamma=0.1)
records = []

print('===> Start training our network .....')
print(nepochs)
for epoch in range(nepochs):
    since = time.time()
    scheduler.step()
    current_lr = scheduler.get_lr()[0]

    train_epoch(net,criterion,optimizer,train_dl,args.lmbd)
    tr_loss, tr_acc = eval(net,criterion,train_dl)
    te_loss, te_acc = eval(net,criterion,test_dl)

    now = time.time()
    records.append((tr_loss,tr_acc,te_loss,te_acc))
    print('[%3d/%d, %.0f seconds]|\t lr=%.2e,  tr_err: %.1e, tr_acc: %.2f |\t te_err: %.1e, te_acc: %.2f'%(
        epoch+1,nepochs,now-since,current_lr,tr_loss,tr_acc,te_loss,te_acc))
print('===> End of training the network -----')

save_model(net,'checkpoints/mnist_width%d_lmbd%.2e_teacc%.2f_tracc%.2f.pkl'%(args.width,args.lmbd,te_acc,tr_acc))
