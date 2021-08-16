import os
import random
from glob import glob
import sys
import numpy as np
import json
from tqdm import tqdm
import pandas as pd
import numpy as np
from ast import literal_eval

import torchtext
from torchtext import data
from torchtext import datasets, vocab

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchtext.data.utils import get_tokenizer

random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

def get_data(tokenization=True, data_dir = '/home/yiyi/MigrTwi/src/hate_speech_detection/processed/GPGC'):


    with open('/home/yiyi/MigrTwi/src/hate_speech_detection/utils/config.json', 'r') as fo:
        config = json.load(fo)

    BATCH_SIZE = config['batch_size']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #### get the spacy tokenizer
    if tokenization:
        tokenizer = get_tokenizer('spacy', language ='en_core_web_sm')

        TEXT = data.Field(sequential=True, tokenize= lambda x: [token for token in tokenizer(x)], lower=True)
    else: 
        TEXT = data.Field(sequential=True, tokenize = lambda x: literal_eval(x) )
        
    LABEL = data.LabelField()



    train  = data.TabularDataset(path=os.path.join(data_dir, 'train.csv'), format='csv', fields=[('text', TEXT), ('label', LABEL)], skip_header=True)
    val  = data.TabularDataset(path=os.path.join(data_dir, 'val.csv'), format='csv', fields=[('text', TEXT), ('label', LABEL)], skip_header=True)
    test  = data.TabularDataset(path=os.path.join(data_dir, 'test.csv'), format='csv', fields=[('text', TEXT), ('label', LABEL)], skip_header=True)


    glove_vectors = vocab.Vectors('/home/yiyi/word_embeds/glove/glove.840B.300d.txt')

    TEXT.build_vocab(train, vectors=glove_vectors)
    LABEL.build_vocab(train)


#     train_it, val_it, test_it = data.Iterator.splits((train, val, test), batch_sizes=(BATCH_SIZE,BATCH_SIZE,BATCH_SIZE), device=device, repeat=False)
    train_it = data.Iterator(train, batch_size=BATCH_SIZE, device=device)
    val_it = data.Iterator(val, batch_size=BATCH_SIZE, device=device)
    test_it = data.Iterator(test, batch_size=BATCH_SIZE, device=device)


    vocab_size = len(TEXT.vocab)
    pretrained_vec = TEXT.vocab.vectors
    
    return config, TEXT, LABEL, train_it, val_it, test_it, vocab_size, pretrained_vec, device
