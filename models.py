import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class TwoLayerNet(nn.Module):
    def __init__(self,input_d,width,output_d,initialize_factor=1):
        super(TwoLayerNet,self).__init__()
        self.fc1 = nn.Linear(input_d,width)
        self.fc2 = nn.Linear(width,output_d,bias=False)
        self.B = self.fc1.weight
        self.C = self.fc1.bias
        self.A = self.fc2.weight

        # initialization
        self.C.data.zero_()
        m,d = self.B.shape
        self.B.data.normal_(0,math.sqrt(2*initialize_factor/d))
        self.A.data.normal_(0,math.sqrt(2*initialize_factor/m))


    def path_norm(self):
        D = torch.cat((self.B,self.C.view(-1,1)),dim=1)
        D = D.norm(p=1,dim=1)
        A = self.A.norm(p=1,dim=0)
        pthnrm = torch.sum(A*D)
        return pthnrm

    def group_norm(self):
        D = torch.cat((self.B,self.C.view(-1,1)),dim=1)
        D = D.pow(2).sum()
        A = self.A.pow(2).sum()

        return math.sqrt(A*D)

    def l2norm(self):
        b = self.B.norm(p=2)
        c = self.C.norm(p=2)
        a = self.A.norm(p=2)

        res = b*b + c*c + a*a
        return res


    def forward(self,x):
        o = x.view(x.size(0),-1)
        o = self.fc1(o)
        o = F.relu(o)
        o = self.fc2(o)
        return o.squeeze()


