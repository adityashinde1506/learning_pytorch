import numpy
import shelve

import torch
import torch.nn as N
import torch.autograd as A
import torch.optim as O
import torch.nn.functional as F

import logging

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
        encoded_data=self.__encode_dataset()
        dataset=[]
        for i in range(0,len(encoded_data)-20):
            frame=encoded_data[i:i+20]
            dataset.append(frame)
        dataset=numpy.array(dataset)
        X,y=numpy.split(dataset,[dataset.shape[1]-1],axis=1)
        return X,y

class LSTMWriter(N.Module):

    def __init__(self,vocab_size,embd_size,hidden_dim,num_hidden_layers):
        super(LSTMWriter,self).__init__()
        self.vocab_size=vocab_size
        self.num_hidden_layers=num_hidden_layers
        self.hidden_dim=hidden_dim
        # Build LSTM layers.
        self.lstm=N.LSTM(input_size=embd_size,hidden_size=hidden_dim,num_layers=num_hidden_layers,batch_first=True)
        # Build word emneddings.
        self.word_embd=N.Embedding(vocab_size,embd_size)
        self.word_predictor=N.Linear(20*19,vocab_size)
        self.hidden=self.init_hidden(num_hidden_layers,hidden_dim)
        self.loss=N.NLLLoss()
        self.optimizer=O.Adam(self.parameters())

    def init_hidden(self,num_layers,hidden_dim):
        return (A.Variable(torch.zeros(num_layers,1,hidden_dim)),A.Variable(torch.zeros(num_layers,1,hidden_dim)))

    def forward(self,sequence):
        embeddings=self.word_embd(sequence)
        lstm_out,self.hidden=self.lstm(embeddings,self.hidden)
        flattened=self.word_predictor(lstm_out.contiguous().view(1000,-1))
        predicted_word=F.log_softmax(flattened)
        return predicted_word

    def partial_fit(self,X,y):
        self.zero_grad()
        self.hidden=self.init_hidden(self.num_hidden_layers,self.hidden_dim)
        out=self.forward(X)
        loss=self.loss(out,y)
        loss.backward()
        self.optimizer.step()
        return loss

    def get_batch(self,X,y):
        dataset=numpy.hstack((X,y))
        numpy.random.shuffle(dataset)
        i=0
        while 1:
            batch=dataset[i:i+1000]
            if batch.shape[0]!=1000:
                i=0
                batch=numpy.vstack((batch,dataset[i:1000-batch.shape[0]]))
            train,target=numpy.split(batch,[batch.shape[1]-1],axis=1)
            yield A.Variable(torch.LongTensor(train)),A.Variable(torch.LongTensor(target).contiguous().view(-1))
            i+=1000

    def fit(self,X,y):
        generator=self.get_batch(X,y)
        for i in range(100):
            train,target=next(generator)
            loss=self.partial_fit(train,target)
            logging.info("Epoch: {} Training Loss: {}".format(i,loss.data[0]))

if __name__=="__main__":
    dataset=Dataset(storage["raw_data"],storage["word_dict"])
    vocab_size=len(storage["word_dict"])
    del storage
    X,y=dataset.get_dataset()
    model=LSTMWriter(vocab_size,30,20,2)
    model.fit(X,y)
