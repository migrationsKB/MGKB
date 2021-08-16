import sys, os
from tqdm import tqdm

import numpy as np
import pandas as pd
from os import path
from gensim.models import KeyedVectors
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import pickle
import json
    
    
class TweetDataset(Dataset):
    def __init__(self, reviews, targets, tokenizer, max_len):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        review = str(self.reviews[item])
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=False,
            return_attention_mask=True,
            return_tensors='pt',
        )

        ### pad the post
        input_ids = pad_sequences(encoding['input_ids'], maxlen=MAX_LEN, dtype=torch.Tensor, truncating="post",
                                  padding="post")
        input_ids = input_ids.astype(dtype='int64')
        input_ids = torch.tensor(input_ids)

        attention_mask = pad_sequences(encoding['attention_mask'], maxlen=MAX_LEN, dtype=torch.Tensor,
                                       truncating="post", padding="post")
        attention_mask = attention_mask.astype(dtype='int64')
        attention_mask = torch.tensor(attention_mask)

        return {
            'review_text': review,
            'input_ids': input_ids,
            'attention_mask': attention_mask.flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }

    
    
