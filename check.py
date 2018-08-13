import argparse
import torch
from models import TwoLayerNet

argparser = argparse.ArgumentParser()
argparser.add_argument('--width',type=int,default=200)
args = argparser.parse_args()

net = TwoLayerNet(32*32*3,args.width,10)
net_state_dict = torch.load('checkpoints/cifar10_width%d.pkl'%(args.width))
net.load_state_dict(net_state_dict)

print(net)
print('path_norm: %.2e'%(net.path_norm()))
print('group_norm: %.2e'%(net.group_norm()))



