import numpy

import logging

logger=logging.getLogger(__name__)

def get_data(filenames):
    _set=set()
    raw_data=""
    for filename in filenames:
        _file=open(filename)
        text=_file.read()
        logger.info("Finished reading {}".format(_file))
        raw_data+=text
        all_chars=_set.union(set(text))
        _file.close()
    char_dict=dict((c,i) for (i,c) in enumerate(all_chars))
    rev_dict=dict((i,c) for (i,c) in enumerate(all_chars))
    logger.debug("Dicts now contain {} characters.".format(len(char_dict)))
    logger.info("Dicts created.")
    return char_dict,rev_dict,numpy.array(list(map(lambda x:char_dict[x],raw_data)))

def make_batches(data,batch_size):
    extra=len(data)%batch_size
    data=data[:len(data)-extra].reshape((batch_size),-1)
    return data

if __name__=="__main__":
    import sys
    cdict,rdict,data=get_data([sys.argv[1]])
    print(len(data))
    print(batch_generator(data,100))
    print(batch_generator(data,1000))
    print(batch_generator(data,1))
