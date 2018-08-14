import time
import os
import argparse
import math
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
from torch.optim import lr_scheduler

from models import TwoLayerNet
from trainer import *
from data import load_mnist

argparser = argparse.ArgumentParser()
argparser.add_argument('--width',type=int,default=1000)
argparser.add_argument('--nepochs',type=int,default=100)
argparser.add_argument('--lr',type=float,default=0.01)
argparser.add_argument('--initialize_factor',type=float,default=1)
argparser.add_argument('--lmbd', type=float, default=0.0)
argparser.add_argument('--nsamples', type=int, default=60000)
argparser.add_argument('--batch_size',type=int,default=100)
argparser.add_argument('--gpuid',default='0')
argparser.add_argument('--weight_decay',type=float,default=0)
argparser.add_argument('--plot',action='store_true')
args = argparser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES']=args.gpuid

# Data Load
train_dl, test_dl = load_mnist(batch_size = args.batch_size,
                                nsamples = args.nsamples,
                                root = './data/mnist',
                                one_hot = True,
                                classes = [0,1])

# Train the model
net = TwoLayerNet(784,args.width,2,args.initialize_factor)
net = net.cuda()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(),
                            lr = args.lr,weight_decay=args.weight_decay)
                            #,momentum=0.9,nesterov=True)

# optimizer = torch.optim.LBFGS(net.parameters(),lr=args.lr,max_iter=10)
scheduler = lr_scheduler.MultiStepLR(optimizer,
                          milestones=[int(args.nepochs*0.7),
                                     int(args.nepochs*0.8),
                                     int(args.nepochs*0.9)],
                          gamma=0.1)
lmbd_base = math.log10(2*784)/args.nsamples/10.0
lmbd = lmbd_base * args.lmbd
records = []

print('lambda=%.2e'%(lmbd_base*args.lmbd))
print('===> Start training our network .....')
for epoch in range(args.nepochs):
    since = time.time()
    scheduler.step()
    current_lr = scheduler.get_lr()[0]

    train_epoch(net,criterion,optimizer,train_dl,lmbd)
    tr_loss, tr_acc = eval(net,criterion,train_dl)
    te_loss, te_acc = eval(net,criterion,test_dl)
    total_loss = tr_loss + lmbd * net.path_norm().item()

    now = time.time()
    records.append((tr_loss,tr_acc,te_loss,te_acc,net.path_norm().item(),total_loss))
    print('[%3d/%d, %.0f secs]| tot_l=%.2e,  tr_err: %.1e, tr_acc: %.2f | te_err: %.1e, te_acc: %.2f, pnorm: %.1e'%(
                epoch+1,args.nepochs,now-since,
                total_loss,tr_loss,tr_acc,te_loss,te_acc,net.path_norm().item()))
print('===> End of training the network -----')
print('the error of objective function is %.2e'%(total_loss))




####################################################
# Store information
####################################################
file_prefix = 'mnist_wdth%d_lmbd%.0e_wd%.0e_lr%.1e_init%.1f_bz%d_totalLoss%.2e'%(
                args.width,args.lmbd,args.weight_decay,args.lr,args.initialize_factor,
                args.batch_size,total_loss)
res = {
   'model_state':net.state_dict(),
   'learning_process':records,
   'nsamples':args.nsamples,
   'weight_decay':args.weight_decay,
   'lambda': args.lmbd,
   'width': args.width,
   'init_var': args.initialize_factor,
   'batch_size': args.batch_size,
   'lr': args.lr
}
with open('checkpoints/%s.pkl'%(file_prefix),'wb') as f:
    pickle.dump(res,f)




if args.plot:
    records = np.asarray(records)
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.semilogy(records[:,0],label='train error')
    plt.semilogy(records[:,2],label='test error')
    plt.legend()

    plt.subplot(1,3,2)
    plt.plot(records[:,1],label='train accuracy')
    plt.plot(records[:,3],label='test accuracy: %.2f'%(te_acc))
    plt.legend()

    plt.subplot(1,3,3)
    plt.semilogy(records[:,4],label='path norm')
    plt.ylim([1,2e6])
    plt.legend()

    plt.savefig(
        'figures/%s_.png'%(file_prefix),bbox_inches='tight'
    )


