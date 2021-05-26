import gzip
import json
import re
import sys
import os
from glob import glob
from collections import OrderedDict, defaultdict

import pandas as pd
import numpy as np
from tqdm import tqdm

sys.path.insert(0, 'MGKB/')

from src.utils.reader import read_gz_file
from src.entity_linking import main_linking
from src.entity_linking.main_linking import load_prerequisites, config, get_entities_from_sentences

print('load prerequisites for blink entity ....')
ner_model, models, args = load_prerequisites(config)

tweets_dict_file = 'entity_linking/relevant_tweets_el_prep.csv'

with open(tweets_dict_file) as file:
    df= pd.read_csv(file)
    

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

sentences= df['prep_el'].tolist()

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
        

output_dir = 'entity_linking/entities_relevant_score/'

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
        
        with open(output_dir+'entities_dict_{}.json'.format(chunk_idx), 'w') as writer:
            json.dump(entities_dict_, writer)

    except Exception:
        print(Exception)
        
    chunk_idx+=1
