import numpy
import shelve

import torch
import torch.nn as N
import torch.autograd as A
import torch.optim as O
import torch.nn.functional as F

import logging

import sys
from generate import *

logging.basicConfig(level=logging.DEBUG)

logger=logging.getLogger(__name__)

storage=shelve.open("data.bin")

SEQ_LEN=200

class Dataset(object):

    def __init__(self,raw_dataset,vocab):
        self.vocab=vocab
        self.raw_data=raw_dataset

    def __encode_dataset(self):
        return list(map(lambda x:self.vocab[x],self.raw_data))

    def get_dataset(self):
        encoded_data=numpy.array(self.__encode_dataset())
        return encoded_data
        

def one_hot_encode(indices,array):
    for i in range(array.shape[0]):
        array[i][indices[i]]=1
    return array

class LSTMWriter(N.Module):
    
    def __init__(self,vocab_size,n_layers=1):
        super(LSTMWriter,self).__init__()
        self.vocab_size=vocab_size
        self.n_layers=n_layers
        self.embedding=N.Embedding(self.vocab_size,30)
        self.lstm=N.GRU(30,100,n_layers,batch_first=True)
        self.linear=N.Linear(100,self.vocab_size)
        self.softmax=N.Sigmoid()
        self.init_weights()

    def init_hidden(self,batch_size=1):
        h=A.Variable(torch.zeros(self.n_layers,batch_size,100))
        c=A.Variable(torch.zeros(self.n_layers,batch_size,200))
        return h

    def init_weights(self):
        self.linear.weight.data.uniform_(-0.01,0.01)
        
    def forward(self,sequence,hidden):
        embedding=self.embedding(sequence.view(1,-1))
        recurrent=embedding
        recurrent,hidden=self.lstm(recurrent.view(1,1,-1),hidden)
        flat=recurrent
        flat=self.linear(flat.view(1,-1))
        return flat,hidden

def get_batches(data):
    i=0
    while 1:
        if i+SEQ_LEN+1>=data.shape[0]:
            break
        train_X=data[i:i+SEQ_LEN]
        train_y=data[i+1:i+SEQ_LEN+1]
        yield numpy.array([train_X]),numpy.array([train_y])
#        print((i,X.shape[0]))
        i+=1

def train_on_sequence(model,inp,target):
    loss=0
    model.zero_grad()
    hidden=model.init_hidden(1)
    for i in range(inp.shape[1]):
        train=A.Variable(torch.from_numpy(inp[:,i]))
        targets=A.Variable(torch.from_numpy(target[:,i]).contiguous().view(-1))
        out,hidden=model.forward(train,A.Variable(hidden.data))
        loss=loss_fn(out,targets)
    loss.backward()
    optimizer.step()
    return loss.data[0]/inp.shape[1]
 

if __name__=="__main__":
    dataset=Dataset(storage["raw_data"],storage["word_dict"])
    vocab_size=len(storage["word_dict"])
    word_dict=storage["word_dict"]
    rev_dict=storage["rev_dict"]
    del storage
    data=dataset.get_dataset()
    loss_fn=N.CrossEntropyLoss()
    seed=data[300:500]
    model=LSTMWriter(vocab_size,1)
    optimizer=O.Adam(model.parameters(),lr=0.001)

    generate_text(seed,model,rev_dict)
    for i in range(20):
        batch_generator=get_batches(data)
        b=0
        while 1:
            total_loss=0
            try:
                _X,_y=next(batch_generator)
            except:
                break
            loss=train_on_sequence(model,_X,_y)
            total_loss+=loss
            b+=1
            if b % 100 ==0:
                logging.info("Epoch :{} Batches:{} Loss :{}".format(i,b,total_loss))
                generate_text(seed,model,rev_dict)
    del data
    #torch.save(model.state_dict(),"model_10_ep.pt")
    #print("Model saved")
    generate_text(seed,model,rev_dict)
