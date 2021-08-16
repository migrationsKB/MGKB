import sys
sys.path.insert(0, '/home/yiyi/MigrTwi')

from itertools import chain
import warnings
warnings.filterwarnings("ignore")

import argparse
import pandas as pd
import torch
from transformers import *
from bert_model import *




parser = argparse.ArgumentParser(description='Entity Linking for Tweets...')
parser.add_argument('--input_file', type=str, default='data/preprocessed/ETM/df_text_preprocessed_tm.csv', help='Input file for sentiment analysis of Tweets...')
parser.add_argument('--output_file', type=str, default='data/preprocessed/bert/df_tm_sentiment.csv', help='Output directory of sentiment analysis...')


args=parser.parse_args()


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
nr_cpu = multiprocessing.cpu_count()
BATCH_SIZE = 16
MAX_LEN = 290

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device:', device)

### for temporal topics text

PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME, do_lower_case=True)
### neutral=1, positive=2, negative=0
model = BertForSequenceClassification.from_pretrained(PRE_TRAINED_MODEL_NAME,  num_labels=3, output_hidden_states=True)

model_path= '../models/Bert/BERT_6.bin'

print('loading model:', model_path)

model.load_state_dict(torch.load(model_path))

model.to(device)



print('loading the dataframe ...')
df = pd.read_csv(args.input_file, index_col=0)

print('df length:', len(df))

# def clean_text(text):
#     text = text.replace('_', ' ')
#     text = pipeline_tp.clean_s(text)
#     return text

df['text_processed']= df['preprocessed_text']

df['sentiment'] = [0 for _ in range(len(df))]


print('create data laoder:')
data_loader =  create_data_loader(df, tokenizer, MAX_LEN, BATCH_SIZE)

model = model.eval()

predictions =[]
# last_hidden_states=[]

count =0 
print('start model evaluations ....')
with torch.no_grad():
    for d in data_loader:
        print('processing data :', count)
        if d['input_ids'].shape[0] < BATCH_SIZE:
            SIZE = d['input_ids'].shape[0]
            input_ids = d["input_ids"].reshape(SIZE, MAX_LEN).to(device)
        else:
            input_ids = d["input_ids"].reshape(BATCH_SIZE, MAX_LEN).to(device)

        attention_mask = d["attention_mask"].to(device)

        outputs = model(input_ids=input_ids, token_type_ids=None, attention_mask=attention_mask)
#         last_hidden_state = outputs[1][0].cpu().detach().numpy()
#         last_hidden_states.append(last_hidden_state)
        
        prediction = torch.argmax(outputs[0], dim=1)
        prediction = prediction.cpu().detach().numpy()
        predictions.append(prediction)

        count+=1
        
predictions = list(chain.from_iterable([list(x) for x in predictions]))

# with open('last_tensor_states_tweets.pickle', 'wb') as handle:
#     pickle.dump(last_hidden_states, handle)


df['pred_sentiment']= predictions
df.to_csv(args.output_file)
