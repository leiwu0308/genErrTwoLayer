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
from data import load_mnist

argparser = argparse.ArgumentParser()
argparser.add_argument('--batch_size',type=int,default=100)
argparser.add_argument('--width',type=int,default=1000)
argparser.add_argument('--nepochs',type=int,default=100)
argparser.add_argument('--lr',type=float,default=0.01)
argparser.add_argument('--lmbd', type=float, default=0.0)
argparser.add_argument('--nsamples', type=int, default=60000)
args = argparser.parse_args()

# Data Load
train_dl, test_dl = load_mnist(batch_size = args.batch_size,
                                nsamples = args.nsamples,
                                root = './data/mnist',
                                one_hot = True)

# Train the model

net = TwoLayerNet(784,args.width,10)
net = net.cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(),
                            lr = args.lr)#,momentum=0.9,nesterov=True)
scheduler = lr_scheduler.MultiStepLR(optimizer,
                          milestones=[int(args.nepochs*0.7),int(args.nepochs*0.9)],
                          gamma=0.1)
lmbd_base = math.log10(2*784)/args.nsamples
records = []

print('===> Start training our network .....')
for epoch in range(args.nepochs):
    since = time.time()
    scheduler.step()
    current_lr = scheduler.get_lr()[0]

    train_epoch(net,criterion,optimizer,train_dl,args.lmbd * lmbd_base)
    tr_loss, tr_acc = eval(net,criterion,train_dl)
    te_loss, te_acc = eval(net,criterion,test_dl)

    now = time.time()
    records.append((tr_loss,tr_acc,te_loss,te_acc))
    print('[%3d/%d, %.0f seconds]|\t lr=%.2e,  tr_err: %.1e, tr_acc: %.2f |\t te_err: %.1e, te_acc: %.2f'%(
        epoch+1,args.nepochs,now-since,current_lr,tr_loss,tr_acc,te_loss,te_acc))
print('===> End of training the network -----')

save_model(net,
        'checkpoints/mnist_width:%d_nsamples:%d_lmbd:%.2e_.pkl'%(args.width,args.nsamples,args.lmbd))
