import time
import torch
from torch.autograd import Variable

def eval(model,criterion,dataloader):
    model.eval()
    loss, acc = 0,0
    for batch_x, batch_y in dataloader:
        batch_x = Variable(batch_x.cuda())
        batch_y = Variable(batch_y.cuda())

        logits = model(batch_x)
        error = criterion(logits,batch_y)
        loss += error.data[0]

        acc += accuracy(logits.data,batch_y.data)

    loss /= len(dataloader)
    acc /= len(dataloader)
    return loss, acc

def train_epoch(model,criterion,optimizer,dataloader,lmbd=0.0):
    model.train()
    for batch_x,batch_y in dataloader:
        batch_x = Variable(batch_x.cuda())
        batch_y = Variable(batch_y.cuda())

        def closure():
            optimizer.zero_grad()
            logits = model(batch_x)
            # print(logits)
            error = criterion(logits,batch_y) + lmbd * model.path_norm()
            error.backward()
            return error

        optimizer.step(closure)


def save_model(net,filename):
    torch.save(net.state_dict(),filename)


def accuracy(logit, target):
    """Computes the precision@k for the specified values of k"""
    batch_size = logit.shape[0]
    if target.ndimension() == 2:
        _, y_true = torch.max(target,1)
    else:
        y_true = target.long().squeeze()
    if logit.ndimension() == 2:
        _, y_pred = torch.max(logit,1)
    else:
        y_pred = (logit>0.5).long().squeeze()

    # print((y_true==y_pred).shape)

    acc = (y_true==y_pred).float().sum()*100.0/batch_size 
    return acc
