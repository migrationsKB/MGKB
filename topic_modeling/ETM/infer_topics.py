import argparse

import torch
import sys
import os
import pandas as pd
import scipy.io
import pickle
import numpy as np
import data
import json
from collections import Counter

parser = argparse.ArgumentParser(description='Infer Topics from Pretrained ETM model')

parser.add_argument('--model_path', type=str, default='')
parser.add_argument('--data_path', type=str, default='/home/yiyi/MigrTwi/082021/data/etm_data/')
parser.add_argument('--batch_size', type=int, default=1000)
parser.add_argument('--num_words', type=int, default=20)
parser.add_argument('--num_topics', type=int, default=50)



args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)



print('loading df...')
df = pd.read_csv('/home/yiyi/MigrTwi/082021/data/preprocessed/df_text_preprocessed_tm.csv', index_col=0)



print('load vocab ...')
vocab_path = os.path.join(args.data_path, 'vocab.pkl')

with open(vocab_path, 'rb') as f:
    vocab = pickle.load(f)
    

### load preprocessed tweets data
print('loading data ...')
token_file = os.path.join(args.data_path, 'data_tokens.mat')
count_file = os.path.join(args.data_path, 'data_counts.mat')
tokens = scipy.io.loadmat(token_file)['tokens'].squeeze()
counts = scipy.io.loadmat(count_file)['counts'].squeeze()

print(tokens.shape, counts.shape)

print('loading model...')
print(args.model_path)

model = torch.load(args.model_path)
model.to(device)


print('beta shape', model.get_beta().shape)

num_docs = len(tokens)
vocab_size = len(vocab)

batch_size =args.batch_size
num_words=args.num_words
num_topics =args.num_topics

word2id = {word:idx for idx, word in enumerate(vocab)}
id2word = {idx:word for idx, word in enumerate(vocab)}


print('starting eval mode ....')

model.eval()

thetaWeights_list=[]
topic2words=dict()

with torch.no_grad():
    beta = model.get_beta()
    for k in range(0,args.num_topics):#topic_indices:
        gamma = beta[k]
        top_words = list(gamma.cpu().numpy().argsort()[-20:][::-1])        
        topic_words = [vocab[a] for a in top_words]
        topic2words[k]=topic_words
        
    ## get most used topics
    indices = torch.tensor(range(num_docs))
    indices = torch.split(indices, batch_size)
    thetaAvg = torch.zeros(1, num_topics).to(device)
    thetaWeightedAvg = torch.zeros(1, num_topics).to(device)
    
    print('theta weighted avg shape: ', thetaWeightedAvg.shape)
    cnt = 0
    for idx, ind in enumerate(indices):
        data_batch = data.get_batch(tokens, counts, ind, vocab_size, device)
        print('data_batch:', len(data_batch))
        sums = data_batch.sum(1).unsqueeze(1)
        cnt += sums.sum(0).squeeze().cpu().numpy()
        normalized_data_batch = data_batch / sums
        # normalized_data_batch = data_batch
        theta, _ = model.get_theta(normalized_data_batch)
        thetaAvg += theta.sum(0).unsqueeze(0) / num_docs
        
        weighed_theta = sums * theta
        print(weighed_theta.shape) # [1000,50]
        
        thetaWeights_list.append(weighed_theta.cpu().detach().numpy())
        weighed_theta_sum= weighed_theta.sum(0).unsqueeze(0)

        print('*'*40)
        thetaWeightedAvg += weighed_theta_sum
        if idx % 100 == 0 and idx > 0:
            print('batch: {}/{}'.format(idx, len(indices)))
    thetaWeightedAvg = thetaWeightedAvg.squeeze().cpu().numpy() / cnt
    print('\nThe 20 most used topics are {}'.format(thetaWeightedAvg.argsort()[::-1][:args.num_words]))

with open(f'/home/yiyi/MigrTwi/082021/data/preprocessed/topicwords_{args.num_topics}.json', 'w') as f:
    json.dump(topic2words, f)
### topic embeddings for the whole data 
thetaWeights = np.concatenate(thetaWeights_list, axis=0)

tokens_list =[]
for sent in tokens.tolist():
    tokens_list.append(sent.tolist()[0])
    
    
print('sorting theta weights ...')
sorted_thetaWeights=[]
for thetaWeight in thetaWeights:
    sorted_thetaWeights.append(thetaWeight.argsort()[::-1].tolist())
    
count=0
indexes = []
indice_chosen=[]
tokens_chosen=[]
prim_topics=[]


for top_nr, inds in zip(sorted_thetaWeights, tokens_list):
    top_words = [topic2words[idx][:3] for idx in top_nr[:3]]
    sents = [id2word[idx] for idx in inds]
    nrs = top_nr[0]
    tops = top_nr[:10]
    count+=1
    
    indice_chosen.append(tops)
    tokens_chosen.append(sents)
    prim_topics.append(nrs)
    
print('count, length of df')
print(count, len(df))    

df['prim_topic']= prim_topics
df['top_topics']= indice_chosen
df.to_csv(f'/home/yiyi/MigrTwi/082021/data/preprocessed/df_text_preprocessed_tm_topic_{args.num_topics}.csv')

print(Counter(prim_topics), len(Counter(prim_topics)))