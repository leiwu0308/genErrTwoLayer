import time
import torch
from torch.autograd import Variable


def eval(model,criterion,dataloader):
    model.eval()
    loss = 0
    accuracy = 0
    for batch_x, batch_y in dataloader:
        batch_x = Variable(batch_x.cuda())
        batch_y = Variable(batch_y.cuda())

        logits = model(batch_x)
        error = criterion(logits,batch_y)
        loss += error.data[0]

        probs,pred_y = logits.data.max(dim=1)
        accuracy += (pred_y==batch_y.data).sum()/batch_y.size(0)

    loss /= len(dataloader)
    accuracy = accuracy*100.0/len(dataloader)
    return loss, accuracy

def train_epoch(model,criterion,optimizer,dataloader,lmbd=0.0):
    model.train()
    for batch_x,batch_y in dataloader:
        batch_x = Variable(batch_x.cuda())
        batch_y = Variable(batch_y.cuda())

        optimizer.zero_grad()
        logits = model(batch_x)
        error = criterion(logits,batch_y) + lmbd * model.path_norm()
        error.backward()
        optimizer.step()

def save_model(net,filename):
    torch.save(net.state_dict(),filename)
