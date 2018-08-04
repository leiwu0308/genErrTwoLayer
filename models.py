import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class TwoLayerNet(nn.Module):
    def __init__(self,input_d,width,output_d):
        super(TwoLayerNet,self).__init__()
        self.fc1 = nn.Linear(input_d,width)
        self.fc2 = nn.Linear(width,output_d,bias=False)

    def path_norm(self):
        B = self.fc1.weight
        C = self.fc1.bias
        A = self.fc2.weight

        D = torch.cat((B,C.view(-1,1)),dim=1)
        D = D.norm(p=1,dim=1)
        A = A.norm(p=1,dim=0)
        pthnrm = torch.sum(A*D)
        return pthnrm

    def group_norm(self):
        B = self.fc1.weight.data
        C = self.fc1.bias.data
        A = self.fc2.weight.data

        D = torch.cat((B,C.view(-1,1)),dim=1)

        A = A.pow(2).sum()
        D = D.pow(2).sum()
        return math.sqrt(A*D)

    def l2norm(self):
        B = self.fc1.weight
        C = self.fc1.bias
        A = self.fc2.weight

        b = B.norm(p=2)
        c = C.norm(p=2)
        a = A.norm(p=2)

        res = b*b + c*c + a*a
        return res


    def forward(self,x):
        o = x.view(x.size(0),-1)
        o = self.fc1(o)
        o = F.relu(o)
        o = self.fc2(o)
        return o


