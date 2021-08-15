import random
import re
import string
import os
import pickle

import numpy as np
import pandas as pd
import preprocessor as p
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import shuffle
from scipy import sparse
from scipy.io import savemat


with open('/home/yiyi/MigrTwi/082021/data/etm_data/vocab.pkl','rb') as f:
    vocab= pickle.load(f)
    
word2id = dict([(w, j) for j, w in enumerate(vocab)])
id2word = dict([(j, w) for j, w in enumerate(vocab)])

df = pd.read_csv('/home/yiyi/MigrTwi/082021/data/preprocessed/df_shuffled_text_preprocessed.csv', index_col=0)

docs = df.preprocessed_text.tolist()
num_docs = len(docs)

print('len of docs:', num_docs)

data = [[word2id[w] for w in docs[idx_d].split() if w in word2id] for idx_d in range(num_docs)]

print('data size: ', len(data))

df['non_empty_data_tm'] = data

df_ = df[df['non_empty_data_tm'].map(lambda d: len(d)>0)]
print('non empty data tm: ',len(df_))
print(df_['non_empty_data_tm'].tolist()[:2])

df_.to_csv('/home/yiyi/MigrTwi/082021/data/preprocessed/df_text_preprocessed_tm.csv')

data = df_['non_empty_data_tm'].tolist()


print('after removing empty data:', len(data))

def create_list_words(in_docs):
    return [x for y in in_docs for x in y]


print('creating lists of words...')
words_data=create_list_words(data)

def create_doc_indices(in_docs):
    aux = [[j for i in range(len(doc))] for j, doc in enumerate(in_docs)]
    return [int(x) for y in aux for x in y]


doc_indices= create_doc_indices(data)
print(len(np.unique(doc_indices)), len(data))

### compressed 
def create_bow(doc_indices, words, n_docs, vocab_size):
    return sparse.coo_matrix(([1] * len(doc_indices), (doc_indices, words)), shape=(n_docs, vocab_size)).tocsr()

print('vocab size:', len(vocab))

num_data = len(data)
print('creating bow representation...')

bow_data= create_bow(doc_indices, words_data, num_data, len(vocab))

print('splitting bow intro token/value pairs and saving to disk...')

def split_bow(bow_in, n_docs):
    indices = [[w for w in bow_in[doc, :].indices] for doc in range(n_docs)]
    counts = [[c for c in bow_in[doc, :].data] for doc in range(n_docs)]
    return indices, counts


bow_tokens, bow_counts = split_bow(bow_data, num_data)

savemat('/home/yiyi/MigrTwi/082021/data/etm_data/' + 'data_tokens.mat', {'tokens': bow_tokens}, do_compression=True)
savemat('/home/yiyi/MigrTwi/082021/data/etm_data/' + 'data_counts.mat', {'counts': bow_counts}, do_compression=True)

print('data ready!')