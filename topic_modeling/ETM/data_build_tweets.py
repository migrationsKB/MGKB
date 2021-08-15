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


# Maximum / minimum document frequency
max_df = 0.7
min_df = 10  # choose desired value for min_df


def build_data(sentence_list):
    # a list of sentences
    #  Create count vectorizer
    print('counting document frequency of words...')
    cvectorizer = CountVectorizer(min_df=min_df, max_df=max_df, stop_words=None)
    cvz = cvectorizer.fit_transform(sentence_list).sign()

    print('building the vocabulary...')
    sum_counts = cvz.sum(axis=0)
    v_size = sum_counts.shape[1]
    sum_counts_np = np.zeros(v_size, dtype=int)
    for v in range(v_size):
        sum_counts_np[v] = sum_counts[0, v]
    # word2id = dict([(w, cvectorizer.vocabulary_.get(w)) for w in cvectorizer.vocabulary_])
    id2word = dict([(cvectorizer.vocabulary_.get(w), w) for w in cvectorizer.vocabulary_])
    del cvectorizer
    print('  initial vocabulary size: {}'.format(v_size))

    # Sort elements in vocabulary
    idx_sort = np.argsort(sum_counts_np)
    vocab_aux = [id2word[idx_sort[cc]] for cc in range(v_size)]

    # Filter out stopwords (if any)
#     vocab_aux = [w for w in vocab_aux if w not in stopwords]
    print('  vocabulary size {}'.format(len(vocab_aux)))

    # Create dictionary and inverse dictionary
    vocab = vocab_aux
    del vocab_aux
    word2id = dict([(w, j) for j, w in enumerate(vocab)])
    id2word = dict([(j, w) for j, w in enumerate(vocab)])

    # Split the data into train/dev/val
    print('tokenizing documents and splitting into train/test/valid...')
    num_docs = len(sentence_list)
    train_size = int(num_docs * 0.8)
    val_size = int(num_docs * 0.05)
    test_size = num_docs - train_size - val_size
    idx_permute = np.random.permutation(train_size + val_size).astype(int)

    # remove words not in train_data
    vocab = list(
        set([w for idx_d in range(train_size) for w in sentence_list[idx_permute[idx_d]].split() if w in word2id]))
    word2id = dict([(w, j) for j, w in enumerate(vocab)])
    id2word = dict([(j, w) for j, w in enumerate(vocab)])
    print('  vocabulary after removing words not in train: {}'.format(len(vocab)))

    # Split in train/test/val
    train_data = [[word2id[w] for w in sentence_list[idx_permute[idx_d]].split() if w in word2id] for idx_d in
                  range(train_size)]
    val_data = [[word2id[w] for w in sentence_list[idx_permute[idx_d + train_size]].split() if w in word2id] for idx_d
                in range(val_size)]
    test_data = [[word2id[w] for w in sentence_list[idx_d + train_size + val_size].split() if w in word2id] for idx_d in
                 range(test_size)]

    print('  number of documents (train): {} [this should be equal to {}]'.format(len(train_data), train_size))
    print('  number of documents (valid): {} [this should be equal to {}]'.format(len(val_data), val_size))
    print('  number of documents (test): {} [this should be equal to {}]'.format(len(test_data), test_size))

    def remove_empty(in_docs):
        return [doc for doc in in_docs if doc != []]

    train_data = remove_empty(train_data)
    val_data = remove_empty(val_data)
    test_data = remove_empty(test_data)

    print('  number of documents (train): {} [compare to before {}]'.format(len(train_data), train_size))
    print('  number of documents (valid): {} [compare to before {}]'.format(len(val_data), val_size))
    print('  number of documents (test): {} [compare to before {}]'.format(len(test_data), test_size))

    # Remove test documents with length=1
    test_data = [doc for doc in test_data if len(doc) > 1]

    # Split test set in 2 halves
    print('splitting test documents in 2 halves...')
    test_data_h1 = [[w for i, w in enumerate(doc) if i <= len(doc) / 2.0 - 1] for doc in test_data]
    test_data_h2 = [[w for i, w in enumerate(doc) if i > len(doc) / 2.0 - 1] for doc in test_data]

    print('test_data_h1 len {}'.format(len(test_data_h1)))
    print('test_data_h1 len {}'.format(len(test_data_h2)))
    # Getting lists of words and doc_indices
    print('creating lists of words...')

    def create_list_words(in_docs):
        return [x for y in in_docs for x in y]

    words_train = create_list_words(train_data)
    words_test = create_list_words(test_data)
    words_test_h1 = create_list_words(test_data_h1)
    words_test_h2 = create_list_words(test_data_h2)
    words_val = create_list_words(val_data)

    print(words_train[:10])
    print('  len(words_tr): ', len(words_train))
    print('  len(words_ts): ', len(words_test))
    print('  len(words_ts_h1): ', len(words_test_h1))
    print('  len(words_ts_h2): ', len(words_test_h2))
    print('  len(words_va): ', len(words_val))

    print('getting doc indices...')

    def create_doc_indices(in_docs):
        aux = [[j for i in range(len(doc))] for j, doc in enumerate(in_docs)]
        return [int(x) for y in aux for x in y]

    doc_indices_train = create_doc_indices(train_data)
    doc_indices_test = create_doc_indices(test_data)
    doc_indices_test_h1 = create_doc_indices(test_data_h1)
    doc_indices_test_h2 = create_doc_indices(test_data_h2)
    doc_indices_val = create_doc_indices(val_data)

    print(
        '  len(np.unique(doc_indices_tr)): {} [this should be {}]'.format(len(np.unique(doc_indices_train)),
                                                                          len(train_data)))
    print(
        '  len(np.unique(doc_indices_ts)): {} [this should be {}]'.format(len(np.unique(doc_indices_test)),
                                                                          len(test_data)))
    print('  len(np.unique(doc_indices_ts_h1)): {} [this should be {}]'.format(len(np.unique(doc_indices_test_h1)),
                                                                               len(test_data_h1)))
    print('  len(np.unique(doc_indices_ts_h2)): {} [this should be {}]'.format(len(np.unique(doc_indices_test_h2)),
                                                                               len(test_data_h2)))
    print(
        '  len(np.unique(doc_indices_va)): {} [this should be {}]'.format(len(np.unique(doc_indices_val)),
                                                                          len(val_data)))
    # Number of documents in each set
    n_train = len(train_data)
    n_test = len(test_data)
    n_test_h1 = len(test_data_h1)
    n_test_h2 = len(test_data_h2)
    n_val = len(val_data)

    del train_data
    del test_data
    del test_data_h1
    del test_data_h2
    del val_data

    def create_bow(doc_indices, words, n_docs, vocab_size):
        return sparse.coo_matrix(([1] * len(doc_indices), (doc_indices, words)), shape=(n_docs, vocab_size)).tocsr()

    print('creating bow representations')
    bow_train = create_bow(doc_indices_train, words_train, n_train, len(vocab))
    bow_test = create_bow(doc_indices_test, words_test, n_test, len(vocab))
    bow_test_h1 = create_bow(doc_indices_test_h1, words_test_h1, n_test_h1, len(vocab))
    bow_test_h2 = create_bow(doc_indices_test_h2, words_test_h2, n_test_h2, len(vocab))
    bow_val = create_bow(doc_indices_val, words_val, n_val, len(vocab))

    del words_train
    del words_test
    del words_test_h1
    del words_test_h2
    del words_val
    del doc_indices_train
    del doc_indices_test
    del doc_indices_test_h1
    del doc_indices_test_h2
    del doc_indices_val

    output_save_path = 'data/etm_data/'
    if not os.path.exists(output_save_path):
        os.mkdir(output_save_path)

    with open(output_save_path + 'vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)

    del vocab
    # Split bow intro token/value pairs
    print('splitting bow intro token/value pairs and saving to disk...')

    def split_bow(bow_in, n_docs):
        indices = [[w for w in bow_in[doc, :].indices] for doc in range(n_docs)]
        counts = [[c for c in bow_in[doc, :].data] for doc in range(n_docs)]
        return indices, counts

    ############# bow train #####################################
    bow_tr_tokens, bow_tr_counts = split_bow(bow_train, n_train)
    savemat(output_save_path + 'bow_tr_tokens.mat', {'tokens': bow_tr_tokens}, do_compression=True)
    savemat(output_save_path + 'bow_tr_counts.mat', {'counts': bow_tr_counts}, do_compression=True)

    del bow_train
    del bow_tr_tokens
    del bow_tr_counts
    ############# bow test #####################################

    bow_ts_tokens, bow_ts_counts = split_bow(bow_test, n_test)
    savemat(output_save_path + 'bow_ts_tokens.mat', {'tokens': bow_ts_tokens}, do_compression=True)
    savemat(output_save_path + 'bow_ts_counts.mat', {'counts': bow_ts_counts}, do_compression=True)
    del bow_test
    del bow_ts_tokens
    del bow_ts_counts

    ################## bow test h1 #############################

    bow_ts_h1_tokens, bow_ts_h1_counts = split_bow(bow_test_h1, n_test_h1)
    savemat(output_save_path + 'bow_ts_h1_tokens.mat', {'tokens': bow_ts_h1_tokens}, do_compression=True)
    savemat(output_save_path + 'bow_ts_h1_counts.mat', {'counts': bow_ts_h1_counts}, do_compression=True)
    del bow_test_h1
    del bow_ts_h1_tokens
    del bow_ts_h1_counts
    ################## bow test h2 #############################

    bow_ts_h2_tokens, bow_ts_h2_counts = split_bow(bow_test_h2, n_test_h2)
    savemat(output_save_path + 'bow_ts_h2_tokens.mat', {'tokens': bow_ts_h2_tokens}, do_compression=True)
    savemat(output_save_path + 'bow_ts_h2_counts.mat', {'counts': bow_ts_h2_counts}, do_compression=True)
    del bow_test_h2
    del bow_ts_h2_tokens
    del bow_ts_h2_counts
    ################## bow val #############################

    bow_val_tokens, bow_val_counts = split_bow(bow_val, n_val)
    savemat(output_save_path + 'bow_va_tokens.mat', {'tokens': bow_val_tokens}, do_compression=True)
    savemat(output_save_path + 'bow_va_counts.mat', {'counts': bow_val_counts}, do_compression=True)
    del bow_val
    del bow_val_tokens
    del bow_val_counts
    print('Data ready !!')
    print('*************')


def read_data(data_path='data/preprocessed/df_geo_text_preprocessed.csv'):
    df = pd.read_csv(data_path, index_col=0)
    df = df.dropna(subset=['preprocessed_text'])
    df = shuffle(df)
    print('size of df:', len(df))
    df.to_csv('data/preprocessed/df_shuffled_text_preprocessed.csv')
    texts = df['preprocessed_text'].tolist()
    return texts


if __name__ == '__main__':
    texts = read_data()
    build_data(texts)
