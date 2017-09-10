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

    def __init__(self,vocab_size,embd_size,hidden_dim,num_hidden_layers,word_dict):
        super(LSTMWriter,self).__init__()
        self.word_dict=word_dict
        self.vocab_size=vocab_size
        self.num_hidden_layers=num_hidden_layers
        self.hidden_dim=hidden_dim
        # Build LSTM layers.
        self.lstm=N.LSTM(input_size=embd_size,hidden_size=hidden_dim,num_layers=num_hidden_layers,batch_first=True)
        # Build word emneddings.
        self.word_embd=N.Embedding(vocab_size,embd_size)
        self.word_predictor=N.Linear(hidden_dim,vocab_size)
        self.hidden=self.init_hidden(num_hidden_layers,hidden_dim,1000)
        self.loss=N.NLLLoss()
        self.optimizer=O.Adam(self.parameters())

    def init_hidden(self,num_layers,hidden_dim,batch_size):
        return (A.Variable(torch.zeros(num_layers,batch_size,hidden_dim)),A.Variable(torch.zeros(num_layers,batch_size,hidden_dim)))

    def forward(self,sequence):
        batch_size=sequence.size(0)
        embeddings=self.word_embd(sequence)
        lstm_out,self.hidden=self.lstm(embeddings,self.hidden)
        flattened=self.word_predictor(lstm_out[:,-1])
        predicted_word=F.log_softmax(flattened)
        return predicted_word

    def partial_fit(self,X,y):
        self.zero_grad()
        self.hidden=self.init_hidden(self.num_hidden_layers,self.hidden_dim,X.size(0))
        out=self.forward(X)
        loss=self.loss(out,y)
        loss.backward()
        self.optimizer.step()
        return loss

    def predict(self,X):
        self.hidden=self.init_hidden(self.num_hidden_layers,self.hidden_dim,1)
        prediction=self.forward(A.Variable(torch.LongTensor(numpy.array([X]))))
        #print("Start")
        #print(prediction.div(10.0).exp().data[0])
        #print(prediction.exp().data[0])
        #print(prediction.div(0.1).exp().data[0])
        return torch.multinomial(prediction.div(0.5).exp(),1)[0].data[0]
        #return prediction.max(1)[1].data[0]

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

    def generate_text(self,seed,len_=100):
        gen_str=" ".join(list(map(lambda x:self.word_dict[x],seed)))
        for i in range(len_):
            prediction=self.predict(seed)
            seed=numpy.append(seed,prediction)
            seed=seed[1:]
            gen_str+=" "+self.word_dict[prediction]
        for line in gen_str.split("<EOS>"):
            print(line)

    def fit(self,X,y):
        generator=self.get_batch(X,y)
        for i in range(200):
            train,target=next(generator)
            loss=self.partial_fit(train,target)
            logging.info("Epoch: {} Training Loss: {}".format(i,loss.data[0]))
            #if i % 10 ==0 and i!=0:
             #   self.generate_text(X[10])

if __name__=="__main__":
    dataset=Dataset(storage["raw_data"],storage["word_dict"])
    vocab_size=len(storage["word_dict"])
    word_dict=storage["word_dict"]
    rev_dict=storage["rev_dict"]
    del storage
    X,y=dataset.get_dataset()
    model=LSTMWriter(vocab_size,30,20,2,rev_dict)
    model.generate_text(X[3],1000)
    model.fit(X,y)
    model.generate_text(X[3],1000)
