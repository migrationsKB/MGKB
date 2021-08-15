import gzip
import json
import re
import sys
import os
import argparse
from glob import glob
from collections import OrderedDict, defaultdict
from ast import literal_eval
import pandas as pd
import numpy as np
from tqdm import tqdm


parser = argparse.ArgumentParser(description='Entity Linking for Tweets...')
parser.add_argument('--input_file', type=str, default='data/preprocessed/df_tm_sentiment_hsd.csv', help='Input file for Entity Linking of Tweets...')
parser.add_argument('--input_dir', type=str, default='data/entities/', help='Ouput directory of Entity Linking...')
parser.add_argument('--output_dir', type=str, default='data/', help='Output directory of Entity Linking...')

args=parser.parse_args()


print('loading the dataframe....')
df = pd.read_csv(args.input_file)

entities_dict =dict()
for file in glob(os.path.join(args.input_dir,'**.json')):
    with open(file) as reader:
        data = json.load(reader)
        data= literal_eval(data)
        for _, item in data.items():
            if 'entities' in item:
                ents = item['entities']
                for ent in ents:
                    idx = ent['id']
                    
                    if idx not in entities_dict:
                        description= ent['description']
                        entity=ent['entity']
                        url = ent['url']
                        entities_dict[idx]={
                            'description': description,
                            'entity': entity,
                            'url':url
                        }
                   
print(f'length of entity dictionary: {len(entities_dict)}')

key_0 =list(entities_dict.keys())[0]
print(entities_dict[key_0])

with open(os.path.join(args.output_dir, 'entities_dict_extracted_20210810.json'), 'w') as file:
    json.dump(entities_dict, file)

tweets_ids = df.id.tolist()

    
#### get the entities for tweets
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
        
chunks_ = list(chunks(tweets_ids, 100))
nrs = [int(re.sub("\D", "", os.path.basename(x))) for x in glob(os.path.join(args.input_dir,'**.json'))]

print(nrs[:10])
print('get entities for tweets....')

tweet2ent=defaultdict(list)

for nr in nrs:
    file = os.path.join(args.input_dir, f'entitiesentities_dict_{nr}.json')
    tweet_id_chunk = chunks_[nr]
    with open(file) as reader:
        data =json.load(reader)
        data = literal_eval(data)
        for id_, item in data.items():
            tweet_id = tweet_id_chunk[int(id_)]
            if 'entities' in item:
                for ent in item['entities']:
                    tweet2ent[tweet_id].append(
                    {
                        "mention": ent['mention'],
                        'id':ent['id'],
                        "start_pos": ent['start_pos'],
                        'score':ent['score']
                    })
    
    print(file)
    


df_ent = pd.DataFrame.from_dict(tweet2ent, orient='index')

print('Combining dfs....')

cols = df_ent.columns

column_names = ['entity_'+str(idx) for idx in cols]
df_ent.rename(columns=dict(zip(cols, column_names)), inplace=True)

joined_df = df.join(df_ent, how='outer', on='id')

print('len of df:', len(df), ', len of merged df: ', len(joined_df))

joined_df.to_csv(os.path.join(args.output_dir, 'df_sentiment_hsd_entities.csv'))