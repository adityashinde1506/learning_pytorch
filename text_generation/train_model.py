from dataset import get_data,make_batches
import sys
import torch.optim as O
import torch.nn as N
import numpy

sys.path.append("/home/adityas/Projects")

from pytorch_utils import trainer
from rnn_model import LSTMWriter
from generate import generate_text

SEQ_LEN=2

filename=["/home/adityas/nltk_data/corpora/gutenberg/carroll-alice.txt"]

cdict,rdict,data=get_data(filename)
data=make_batches(data,100)

def get_batches(data):
    i=0
    while 1:
        if i+SEQ_LEN+1>data.shape[0]:
            break
        train_X=data[:,i:i+SEQ_LEN]
        train_y=data[:,i+1:i+SEQ_LEN+1]
        yield train_X,train_y
        i+=1

model=LSTMWriter(len(cdict),seq_len=SEQ_LEN)
generate_text(numpy.array([data[0][:SEQ_LEN]]),model,rdict)
rec_trainer=trainer.RecurrentTrainer(model,O.Adam(model.parameters(),lr=0.001))
rec_trainer.train(data,get_batches,N.CrossEntropyLoss(),print_every=50)
#print()
generate_text(numpy.array([data[0][:SEQ_LEN]]),model,rdict)
