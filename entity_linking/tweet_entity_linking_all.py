import gzip
import json
import re
import sys
import os
import argparse
from glob import glob
from collections import OrderedDict, defaultdict

import pandas as pd
import numpy as np
from tqdm import tqdm

sys.path.insert(0, '/home/yiyi/MigrTwi')

from utils.reader import read_gz_file
from entity_linking import main_linking
from entity_linking.main_linking import load_prerequisites, config, get_entities_from_sentences


parser = argparse.ArgumentParser(description='Entity Linking for Tweets...')
parser.add_argument('--input_file', type=str, default='data/preprocessed/df_text_preprocessed_tm.csv', help='Input file for Entity Linking of Tweets...')
parser.add_argument('--output_dir', type=str, default='data/entities', help='Ouput directory of Entity Linking...')

argss=parser.parse_args()


print('load prerequisites for blink entity ....')
ner_model, models, args = load_prerequisites(config)


print(f'Reading input file {argss.input_file} ...')
df = pd.read_csv(argss.input_file, index_col=0)
    

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

sentences= df['text'].tolist()

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
        

def get_latest_nr(output_dir):
    nrs = [int(re.sub("\D", "", x)) for x in glob(output_dir+'**.json')]
    latest = np.max(nrs)
    return latest

# latest_nr = get_latest_nr(output_dir)

chunks_ = list(chunks(sentences, 100))
print(f'there are {len(chunks_)} batches')

chunk_idx =0 
for chunk in tqdm(chunks_):
    print(f'Processing the {chunk_idx}th batch....')
    print('get entities from sentences....')
    try:
        entities_dict = get_entities_from_sentences(chunk,ner_model, models, args)
        entities_dict_ = json.dumps(entities_dict, cls=NpEncoder)
        
        print('write the entities dict to ...','entities_dict{}.json'.format(chunk_idx) )
        
        with open(argss.output_dir+'entities_dict_{}.json'.format(chunk_idx), 'w') as writer:
            json.dump(entities_dict_, writer)

    except Exception:
        print(Exception)
        
    chunk_idx+=1
