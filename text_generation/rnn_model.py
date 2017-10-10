import numpy

import torch
import torch.nn as N
import torch.autograd as A
import torch.optim as O
import torch.nn.functional as F

import logging

logging.basicConfig(level=logging.DEBUG)

logger=logging.getLogger(__name__)

class LSTMWriter(N.Module):
    
    def __init__(self,vocab_size,n_layers=1,seq_len=5):
        super(LSTMWriter,self).__init__()
        self.vocab_size=vocab_size
        self.n_layers=n_layers
        self.seq_len=seq_len
        self.embedding=N.Embedding(self.vocab_size,30)
        self.lstm=N.GRU(30,100,n_layers,batch_first=True)
        self.linear=N.Linear(100,self.vocab_size)
        self.softmax=N.Sigmoid()
        self.init_weights()
        logger.info("NN initialised.")

    def init_hidden(self,batch_size=1):
        h=A.Variable(torch.zeros(self.n_layers,batch_size,100))
        c=A.Variable(torch.zeros(self.n_layers,batch_size,200))
        return h

    def init_weights(self):
        self.linear.weight.data.uniform_(-0.01,0.01)
        
    def forward(self,sequence,hidden):
        embedding=self.embedding(sequence)
        recurrent=embedding.contiguous().view(-1,self.seq_len,30)
        recurrent,hidden=self.lstm(recurrent,hidden)
        flat=recurrent.contiguous().view(-1,100)
        flat=self.linear(flat)
        return flat,hidden
