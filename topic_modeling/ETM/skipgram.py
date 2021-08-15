import gensim
import pickle
import os
import pandas as pd
import numpy as np
import argparse
import re
import string
import preprocessor as p

from nltk.stem import WordNetLemmatizer

wordnet_lemmatizer = WordNetLemmatizer()

parser = argparse.ArgumentParser(description='The Embedded Topic Model')

### data and file related arguments
parser.add_argument('--data_file', type=str, default='/home/yiyi/MigrTwi/082021/data/preprocessed/df_shuffled_text_preprocessed.csv', help='a .txt file containing the corpus')
parser.add_argument('--emb_file', type=str, default='/home/yiyi/MigrTwi/082021/data/etm_data/embeddings.txt', help='file to save the word embeddings')
parser.add_argument('--dim_rho', type=int, default=300, help='dimensionality of the word embeddings')
parser.add_argument('--min_count', type=int, default=2, help='minimum term frequency (to define the vocabulary)')
parser.add_argument('--sg', type=int, default=1, help='whether to use skip-gram')
parser.add_argument('--workers', type=int, default=56, help='number of CPU cores')
parser.add_argument('--negative_samples', type=int, default=10, help='number of negative samples')
parser.add_argument('--window_size', type=int, default=4, help='window size to determine context')
parser.add_argument('--iters', type=int, default=20, help='number of iterationst')

args = parser.parse_args()

# Class for a memory-friendly iterator over the dataset
class MySentences(object):
    def __init__(self, filename):
        self.filename = filename
 
    def __iter__(self):
        for line in open(self.filename):
            yield line.split()

# Gensim code to obtain the embeddings

pattern = r'''[\w']+|[.,!?;-~{}`´<=>:/@*()&'$%#"]'''

punctuations = string.punctuation.replace('_', '').replace('#', '')
p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.MENTION, p.OPT.RESERVED,
              p.OPT.SMILEY, p.OPT.NUMBER)
def load_stopwords():
    # Read stopwords
    with open('ETM/scripts/stops.txt', 'r') as f:
        stops = f.read().split('\n')
    return stops


def contains_punctuation(w):
    return any(char in punctuations for char in w)


def contains_numeric(w):
    return any(char.isdigit() for char in w)


def preprocessing_text(text):
    text = text.replace('\n', '').replace('\r', '')
    text = p.clean(text)

    tokens = re.findall(pattern, text)
    tokens = [token.replace('_', '') if token.startswith('_') or token.endswith('_') else token for token in tokens]
    tokens = [w.lower() for w in tokens if not contains_punctuation(w)]
    tokens = [w for w in tokens if not contains_numeric(w)]
    tokens = [wordnet_lemmatizer.lemmatize(w)for w in tokens]

    tokens = [w for w in tokens if len(w) > 1]
    return tokens

def read_data(data_path=args.data_file, preprocessing=False):
    df = pd.read_csv(data_path)
    texts = df.preprocessed_text.tolist()
    print('processing texts.....', preprocessing)
    
    if preprocessing:
        processed= [preprocessing_text(text) for text in texts]

        return processed
    else:
        return texts

sentences = read_data()  # a memory-friendly iterator
print('start training skipgram word embeddings....')
model = gensim.models.Word2Vec(sentences, min_count=args.min_count, sg=args.sg, size=args.dim_rho, 
    iter=args.iters, workers=args.workers, negative=args.negative_samples, window=args.window_size)

# Write the embeddings to a file
with open(args.emb_file, 'w') as f:
    for v in list(model.wv.vocab):
        vec = list(model.wv.__getitem__(v))
        f.write(v + ' ')
        vec_str = ['%.9f' % val for val in vec]
        vec_str = " ".join(vec_str)
        f.write(vec_str + '\n')


