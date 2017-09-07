#!/usr/bin/env python3

import logging
import re
import os
import pickle
import shelve

from functools import *

logging.basicConfig(level=logging.DEBUG)

logger=logging.getLogger(__name__)

obj_file=shelve.open("data.bin")

class Vocab(object):

    def __init__(self):
        self.vocab=set(["<UNK>"])

    def create_index(self):
        enum=enumerate(self.vocab)
        logger.info("Building word dict.")
        self.word_dict=dict((word,i) for i,word in enum)
        logger.info("Building reverse dict.")
        self.rev_dict=dict((i,word) for i,word in enum)

    def update(self,new_set):
        logger.info("Updating vocab.")
        self.vocab=self.vocab.union(new_set)
        logger.info("New vocab has {} words".format(len(self.vocab)))

    def encode(self,word):
        try:
            return self.word_dict[word]
        except:
            return self.word_dict["<UNK>"]

    def encode_dataset(self,dataset):
        return list(map(lambda x:self.encode(x),dataset))


DATA_DIR="/home/adityas/nltk_data/corpora/gutenberg/"

files=os.listdir(DATA_DIR)[:1]
logger.info("Data files are {}".format(", ".join(files)))
files=map(lambda x:DATA_DIR+x,files)

SPLIT_PATTERN=r"[.?!;]+"

vocab=Vocab()
raw_dataset=[]

for _file in files:
    logger.info("Reading {}".format(_file))
    try:
        _file_ptr=open(_file,encoding="ASCII")
        raw_text=re.sub(r"[\s\(\)\[\]]+"," ",_file_ptr.read().lower())
        sentences=re.split(SPLIT_PATTERN,raw_text)
        logger.info("Splitting {} into words.".format(_file))
        words=reduce(lambda x,y:x+y,map(lambda x:re.findall(r"\w+",x)+["<EOS>"],sentences))
        vocab.update(set(words))
        raw_dataset+=words
    except Exception as e:
        logger.error("Could not read {}! Reason: {}".format(_file,e))

vocab.create_index()
logger.info("Done building dataset. Pickling stuff.")
obj_file["vocab"]=vocab
obj_file["raw_data"]=raw_dataset
obj_file.close()
