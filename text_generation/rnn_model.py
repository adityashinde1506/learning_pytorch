import numpy
import shelve

import torch
import torch.nn as N
import torch.autograd as A
import torch.optim as O
import torch.nn.functional as F

import logging

import sys

logging.basicConfig(level=logging.DEBUG)

logger=logging.getLogger(__name__)

storage=shelve.open("data.bin")

class Dataset(object):

    def __init__(self,raw_dataset,vocab):
        self.vocab=vocab
        self.raw_data=raw_dataset

    def __encode_dataset(self):
        return list(map(lambda x:self.vocab[x],self.raw_data))

    def get_dataset(self):
        encoded_data=numpy.array(self.__encode_dataset()[:-483]).reshape((1000,-1))
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
        self.lstm=N.GRU(30,30,n_layers,batch_first=True)
        self.dropout=N.Dropout(0.1)
        self.linear=N.Linear(30,self.vocab_size)
        self.relu=N.ReLU()
        self.linear2=N.Linear(self.vocab_size,self.vocab_size)
        self.softmax=N.Softmax()

    def init_hidden(self,batch_size=1):
        h=A.Variable(torch.zeros(self.n_layers,batch_size,30))
        c=A.Variable(torch.zeros(self.n_layers,batch_size,10))
        return h
        
    def forward(self,sequence,hidden):
        embedding=self.embedding(sequence)
        recurrent=embedding
        #for i in range(self.n_layers):
        recurrent,hidden=self.lstm(recurrent,hidden)
        flat=self.linear(recurrent.contiguous().view(recurrent.size(0)*recurrent.size(1),recurrent.size(2)))
        #flat=self.softmax(flattened)
        #flattened=recurrent[:,-1]
        #flat=self.linear(flattened)
        flat=self.linear2(self.dropout(self.relu(flat)))
        flat=flat.view(recurrent.size(0),recurrent.size(1),flat.size(1))
        return flat,hidden

def get_batches(data):
    i=0
    while 1:
        if i+11>=data.shape[1]:
            break
        train_X=data[:,i:i+50]
        train_y=data[:,i+1:i+50+1]
        yield train_X,train_y.squeeze()
#        print((i,X.shape[0]))
        i+=1

def generate_text(X,model):
    fp=open("generated_text","a")
    print("Priming the network.")
    hidden=model.init_hidden(1)
    i =0
    while (i+50) < X.shape[0]:
        #print("Priming for {}".format(i))
        input_=numpy.array([X[i:i+50]])
        _,hidden=model.forward(A.Variable(torch.from_numpy(input_)),hidden)
        i+=1
    gen_str=""
    print("Network primed")
    for i in range(5000):
        out,hidden=model.forward(A.Variable(torch.from_numpy(input_)),hidden)
        out=out[:,-1].exp().data
        #print(out)
        #sys.exit()
        char=torch.multinomial(out,1)[0][0]
        fp.write(rev_dict[char])
        fp.flush()
        input_=numpy.append(input_.squeeze(),char)
        input_=numpy.array([input_[1:]],dtype=numpy.long)
    fp.flush()
    fp.close()

if __name__=="__main__":
    dataset=Dataset(storage["raw_data"],storage["word_dict"])
    vocab_size=len(storage["word_dict"])
    word_dict=storage["word_dict"]
    rev_dict=storage["rev_dict"]
    del storage
    data=dataset.get_dataset()
    loss_fn=N.CrossEntropyLoss()
    seed=data[0][3:300]

    model=LSTMWriter(vocab_size,2)
    optimizer=O.Adam(model.parameters(),lr=0.1)

    for i in range(20):
        hidden=model.init_hidden(1000)
        batch_generator=get_batches(data[:,:10000])
        total_loss=0
        b=1
        while b:
            model.zero_grad()
            try:
                _X,_y=next(batch_generator)
            except:
                break
            train=A.Variable(torch.from_numpy(_X))
            targets=A.Variable(torch.from_numpy(_y).contiguous().view(-1))
            out,hidden=model.forward(train,A.Variable(hidden.data))
            loss=loss_fn(out.contiguous().view(-1,vocab_size),targets)
            loss.backward()
            optimizer.step()
            total_loss+=loss.data[0]
            #del data
            #del train
            #del targets
            #generate_text(seed,model)
            if b % 100 ==0:
                logging.info("Epoch :{} Batches :{} Loss :{}".format(i,b,total_loss))
                total_loss=0
            b+=1
    del data
    generate_text(seed,model)
