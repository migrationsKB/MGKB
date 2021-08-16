import json
import pandas as pd
import numpy as np
from tqdm import tqdm

from pandarallel import pandarallel
import torch
import sys
sys.path.insert(0, '/home/yiyi/MigrTwi/')

from hate_speech_detection.models.models_fusion import *
from hate_speech_detection.utils.build_vocab import *
from hate_speech_detection.utils.utils import *

tokenization=True

config ={'emb_dim': 300,
 'nlayers': 2,
 'bidir': True,
 'dropout': 0.3,
 'hidden_dim': 100,
 'epochs': 10,
 'gpu': True,
 'batch_size': 16,
 'rnn_layer': 'lstm',
 'out_dim':3,
 'has_attn': True}

data_dir ='/home/yiyi/MigrTwi/src/hate_speech_detection/processed/hateXplain_no_emoji'
tokenization=True

config, TEXT, LABEL, train_it, val_it, test_it, vocab_size, pretrained_vec, device = get_data(tokenization=True, data_dir= data_dir)

model = cnnBRNN(vocab_size, config, trainable=True, vec= pretrained_vec)


model_path = '/home/yiyi/MigrTwi/src/hate_speech_detection/hsd_models_cnnblstm/model_epoch_1.pth'

model.to(device)


df = pd.read_csv('/home/yiyi/MigrTwi/082021/data/preprocessed/df_tm_sentiment.csv', index_col=0)
texts = df['text_processed'].tolist()

df_ = df[['text_processed']]
df_.rename(columns= {'text_processed':'text'}, inplace=True)
df_['label']=[0 for x in range(len(df_))]


df_.to_csv('/home/yiyi/MigrTwi/082021/data/preprocessed/processed_for_hsd.csv', index=False)


tweets  = data.TabularDataset(path=os.path.join('/home/yiyi/MigrTwi/082021/data/preprocessed/', 'processed_for_hsd.csv'), format='csv', fields=[('text', TEXT), ('label', LABEL)], skip_header=True)
tweets_it = data.Iterator(tweets, batch_size=config['batch_size'], shuffle=False, device=device)


model.eval()
predictions= []

with torch.no_grad():
    for b in tqdm(tweets_it):
        seq, _ = b.text, b.label
        pred = model(seq)
        pred = pred.argmax(dim=1, keepdim=True)
        pred = pred.cpu().detach().numpy()
        predictions.append(pred)
        
prediction = list(chain.from_iterable(predictions))
        
df['hatespeech_pred']= prediction
print(df['hatespeech_pred'].value_counts())

df.to_csv('/home/yiyi/MigrTwi/082021/data/preprocessed/df_tm_sentiment_hsd.csv')