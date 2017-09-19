import numpy
import shelve

import torch
import torch.nn as N
import torch.autograd as A
import torch.optim as O
import torch.nn.functional as F
import sys

from rnn_model import *

storage=shelve.open("data.bin")

def generate_text(X,model,rev_dict):
    fp=open("generated_text","a")
    print("Priming the network.")
    hidden=model.init_hidden(1)
    i =0
    while i < X.shape[0]:
        #print("Priming for {}".format(i))
        input_=numpy.array([X[i]])
        _,hidden=model.forward(A.Variable(torch.from_numpy(input_)),hidden)
        i+=1
    #gen_str="".join(list(map(lambda x:rev_dict[int(x)],X.squeeze())))
    print("Network primed")
    input_=numpy.array([X[-1]])
    gen_str=""+rev_dict[X[-1]]
    for i in range(1000):
        out,hidden=model.forward(A.Variable(torch.from_numpy(input_)),hidden)
        out_=out.data.view(-1).div(0.6).exp()
        char=torch.multinomial(out_,1)[0]
        gen_str+=rev_dict[char]
        input_=numpy.array([char])
    print(gen_str)
    fp.flush()
    fp.close()

if __name__=="__main__":
    dataset=Dataset(storage["raw_data"],storage["word_dict"])
    data=dataset.get_dataset()
    seed=data[0][10:300]

    model=LSTMWriter(85,1)
    model.load_state_dict(torch.load("model_10_ep.pt"))
    print(model)
    generate_text(seed,model,storage["rev_dict"])
