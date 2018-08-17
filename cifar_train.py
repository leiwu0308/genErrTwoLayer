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
from data import load_cifar


def train_model(args, net, train_dl, test_dl, device, lmbd_base):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr=args.lr, weight_decay=args.weight_decay)

    scheduler = lr_scheduler.MultiStepLR(optimizer,
                                         milestones=[int(args.nepochs*0.7),
                                                     int(args.nepochs*0.8),
                                                     int(args.nepochs*0.9)],
                                         gamma=0.1)
    lmbd = lmbd_base * args.lmbd
    records = []

    print('lambda = %.2e' % (lmbd))
    print('===> Start training our network .....')
    for epoch in range(args.nepochs):
        since = time.time()
        scheduler.step()
        current_lr = scheduler.get_lr()[0]

        train_epoch(net, criterion, optimizer, train_dl, device, lmbd)

        tr_err, tr_acc = eval(net, criterion, train_dl, device)
        total_err = tr_err + lmbd * net.path_norm().item()

        if args.watch_lp or epoch == args.nepochs - 1:
            te_err, te_acc = eval(net, criterion, test_dl, device)
        else:
            te_err, te_acc = -1, -1

        now = time.time()
        records.append((tr_err, tr_acc, te_err, te_acc, net.path_norm().item(), total_err))
        print('[%3d/%d, %.0f s]| tot_E=%.2e,  tr_E: %.1e, tr_A: %.2f| te_E: %.1e, te_A: %.2f, pnorm: %.1e' % (
            epoch+1, args.nepochs, now-since, total_err, tr_err, tr_acc, te_err,
            te_acc, net.path_norm().item()))
    print('===> End of training the network -----')
    print('the error of objective function is %.2e' % (total_err))

    res = (tr_err, tr_acc, te_err, te_acc,
           net.path_norm().item(), net.l2norm().item())

    return res, records


def watch_learning_process(records, file_prefix):
    records = np.asarray(records)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.semilogy(records[:, 0], label='train error')
    plt.semilogy(records[:, 2], label='test error')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(records[:, 1], label='train accuracy')
    plt.plot(records[:, 3], label='test accuracy: %.2f' % (te_acc))
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.semilogy(records[:, 4], label='path norm')
    plt.ylim([1, 2e6])
    plt.legend()

    plt.savefig(
        'figures/%s_.png' % (file_prefix), bbox_inches='tight'
    )


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--nsamples', type=int, default=60000)
    argparser.add_argument('--batch_size', type=int, default=100)

    argparser.add_argument('--width', type=int, default=1000)
    argparser.add_argument('--lmbd', type=float, default=0.0)
    argparser.add_argument('--weight_decay', type=float, default=0)
    argparser.add_argument('--init_fac', type=float, default=1)
    argparser.add_argument('--nepochs', type=int, default=100)
    argparser.add_argument('--lr', type=float, default=0.01)
    argparser.add_argument('--ntries', type=int, default=1)
    argparser.add_argument('--watch_lp', action='store_true')
    argparser.add_argument('--device', default='cuda')
    args = argparser.parse_args()
    device = torch.device(args.device)

    # Data Load
    train_dl, test_dl = load_cifar(batch_size=args.batch_size,
                                   nsamples=args.nsamples,
                                   root='./data/cifar10',
                                   one_hot=False,
                                   nclasses=2)

    # repeat experiments of n times
    res_t = []
    for _ in range(args.ntries):
        net = TwoLayerNet(3*32*32, args.width, 1, args.init_fac).to(device)

        lmbd_base = math.log10(3*32*32)/args.nsamples
        res, _ = train_model(args, net, train_dl, test_dl, device, lmbd_base)
        res_t.append(res)

    tr_err, tr_acc, te_err, te_acc, pnorm, l2norm = zip(*res_t)

    res = {
        'dataset': 'mnist',
        'nsamples': args.nsamples,
        'weight_decay': args.weight_decay,
        'lambda': args.lmbd,
        'width': args.width,
        'init_fac': args.init_fac,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'train_error': tr_err,
        'train_accuracy': tr_acc,
        'test_error': te_err,
        'test_accuracy': te_acc,
        'path_norm': pnorm,
        'l2norm': l2norm
    }

    file_prefix = 'cifar10_wdth%d_lmbd%.0e_wd%.0e_lr%.1e_init%.1f_bz%d' % (
                  args.width, args.lmbd, args.weight_decay, args.lr, args.init_fac,
                  args.batch_size)

    if args.watch_lp:
        watch_learning_process(records, file_prefix)

    with open('checkpoints/%s_.pkl' % (file_prefix), 'wb') as f:
        pickle.dump(res, f)

if __name__ == '__main__':
    main()
