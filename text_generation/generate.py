import numpy
import shelve

import torch
import torch.nn as N
import torch.autograd as A
import torch.optim as O
import torch.nn.functional as F
import sys

from rnn_model import *


def generate_text(X,model,rev_dict):
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
    input_=X
    gen_str=""+"".join(map(lambda x:rev_dict[x],X.squeeze()))
    for i in range(5000):
#        print(input_)
        out,hidden=model.forward(A.Variable(torch.from_numpy(input_)),hidden)
#        print(out)
        out_=out.data[-1,:].div(0.8).exp()
        char=torch.multinomial(out_,1)[0]
#        print(char)
        gen_str+=rev_dict[char]
        input_=numpy.array([numpy.append(input_[0][1:],char)])
    print(gen_str)

if __name__=="__main__":
    dataset=Dataset(storage["raw_data"],storage["word_dict"])
    data=dataset.get_dataset()
    seed=data[0][10:300]

    model=LSTMWriter(85,1)
    model.load_state_dict(torch.load("model_10_ep.pt"))
    print(model)
    generate_text(seed,model,storage["rev_dict"])
