import multiprocessing
from collections import defaultdict
import json
import os 

import numpy as np
import pandas as pd
from pandarallel import pandarallel
import torch
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.utils import shuffle
from transformers import XLNetTokenizer, XLNetForSequenceClassification, AdamW, \
    get_linear_schedule_with_warmup
from torch import nn
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import Dataset, DataLoader


pandarallel.initialize()
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
nr_cpu = multiprocessing.cpu_count()
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
MAX_LEN = 290
print('torch cuda available: ', torch.cuda.is_available())


##### xlnet model and tokenizer #####
def load_model_tokenizer():
    """
    Load model and tokenizer
    """
    PRE_TRAINED_MODEL_NAME = 'xlnet-base-cased'
    tokenizer = XLNetTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME, do_lower_case=True)
    ### neutral=1, positive=2, negative=0
    model = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased',  num_labels=3)
    model = model.to(device)
    return tokenizer, model


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


def create_data_loader(df, tokenizer, max_len, batch_size):
    """

    :param df:
    :param tokenizer:
    :param max_len:
    :param batch_size:
    :return:
    """
    
    df = df.dropna(subset=['text_processed'])
    sentences = df.text_processed.values
    sentences_ =  [sentence + " [SEP] [CLS]" for sentence in sentences]
    df["sents"]= sentences_
    
    ds = TweetDataset(
        reviews=df.sents.to_numpy(),
        targets=df.sentiment.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=nr_cpu
    )


def data_loaders(df_train, df_val, df_test, tokenizer):
    # df = shuffle(df)
    # df_train, df_test = train_test_split(df, test_size=0.3, random_state=101)
    # df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=101)
    print('=====stats about data=================')
    print('validation data: ', df_val.sentiment.value_counts())
    print('test data: ', df_test.sentiment.value_counts())
    print('train data: ', df_train.sentiment.value_counts())
    print('data shapes:', df_train.shape, df_val.shape, df_test.shape)


    train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
    val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
    test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)
    return train_data_loader, val_data_loader, test_data_loader


def setting_hyperparameters(model, LEN_train_data):
    EPOCHS = 8  # recommend 2 to 4
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5, eps=1e-6)

    total_steps = LEN_train_data * EPOCHS

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    return EPOCHS, optimizer, scheduler


def train_epoch(model, data_loader, optimizer, device, scheduler, n_examples):
    model = model.train()
    losses = []
    acc = 0
    prec = 0
    rec = 0
    f1 = 0

    prec_micro = 0
    rec_micro = 0
    f1_micro = 0

    counter = 0

    for d in data_loader:
        if d['input_ids'].shape[0] < BATCH_SIZE:
            SIZE = d['input_ids'].shape[0]
            input_ids = d["input_ids"].reshape(SIZE, MAX_LEN).to(device)
        else:
            input_ids = d["input_ids"].reshape(BATCH_SIZE, MAX_LEN).to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)

        outputs = model(input_ids=input_ids, token_type_ids=None, attention_mask=attention_mask, labels=targets)
        loss = outputs[0]
        logits = outputs[1]

        _, prediction = torch.max(logits, dim=1)
        targets = targets.cpu().detach().numpy()
        prediction = prediction.cpu().detach().numpy()
        accuracy = metrics.accuracy_score(targets, prediction)
        precision = metrics.precision_score(targets, prediction,labels=np.unique(prediction), average='macro')
        recall = metrics.recall_score(targets, prediction,labels=np.unique(prediction), average='macro')
        f1_score = metrics.f1_score(targets, prediction,labels=np.unique(prediction), average='macro')

        precision_micro = metrics.precision_score(targets, prediction, labels=np.unique(prediction), average='micro')
        recall_micro = metrics.recall_score(targets, prediction, labels=np.unique(prediction), average='micro')
        f1_score_micro = metrics.f1_score(targets, prediction, labels=np.unique(prediction), average='micro')

        prec_micro += precision_micro
        rec_micro += recall_micro
        f1_micro += f1_score_micro

        prec += precision
        rec += recall
        f1 += f1_score

        acc += accuracy

        losses.append(loss.item())

        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        counter = counter + 1
        print('train counter', counter, end='\r')
    
    print('accuracy:', acc/counter)
    print('loss:', np.mean(losses))
    print('macro f1 score:', f1/counter)
    print('macro precision:', prec/counter)
    print('macro recall:', rec/counter)
    print('micro f1 score:', f1_micro/counter)
    print('micro precision:', prec_micro/counter)
    print('micro recall', rec_micro/counter)

    return acc / counter, prec/counter, rec/counter, f1/counter, prec_micro/counter, rec_micro/counter, \
        f1_micro/counter, np.mean(losses)


def eval_model(model, data_loader, device, n_examples):
    model = model.eval()
    losses = []
    acc = 0
    prec = 0
    rec = 0
    f1 = 0

    prec_micro = 0
    rec_micro = 0
    f1_micro = 0
    counter = 0

    with torch.no_grad():
        for d in data_loader:
            if d['input_ids'].shape[0] < BATCH_SIZE:
                SIZE = d['input_ids'].shape[0]
                input_ids = d["input_ids"].reshape(SIZE, MAX_LEN).to(device)
            else:
                input_ids = d["input_ids"].reshape(BATCH_SIZE, MAX_LEN).to(device)

            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)

            outputs = model(input_ids=input_ids, token_type_ids=None, attention_mask=attention_mask, labels=targets)
            loss = outputs[0]
            logits = outputs[1]

            _, prediction = torch.max(logits, dim=1)
            targets = targets.cpu().detach().numpy()
            prediction = prediction.cpu().detach().numpy()

            accuracy = metrics.accuracy_score(targets, prediction)
            precision = metrics.precision_score(targets, prediction,labels=np.unique(prediction),average='macro')
            recall = metrics.recall_score(targets, prediction, labels=np.unique(prediction), average='macro')
            f1_score = metrics.f1_score(targets, prediction,labels=np.unique(prediction), average='macro')

            precision_micro = metrics.precision_score(targets, prediction, labels=np.unique(prediction), average='micro')
            recall_micro = metrics.recall_score(targets, prediction, labels=np.unique(prediction), average='micro')
            f1_score_micro = metrics.f1_score(targets, prediction, labels=np.unique(prediction), average='micro')

            prec_micro += precision_micro
            rec_micro += recall_micro
            f1_micro += f1_score_micro

            prec += precision
            rec += recall
            f1 += f1_score
            acc += accuracy
            losses.append(loss.item())
            counter += 1

    return acc / counter, prec/counter, rec/counter, f1/counter, prec_micro/counter, rec_micro/counter, \
        f1_micro/counter, np.mean(losses)

def train(dataset, output_dir):
    data_dir ='sentiment_analysis/preprocessed_data/pipeline_tp_2/'+dataset
    df_val = pd.read_csv(data_dir + '/val.csv')
    df_test = pd.read_csv(data_dir + '/test.csv')
    df_train = pd.read_csv(data_dir + '/train.csv')
    
    tokenizer, model = load_model_tokenizer()
    train_data_loader, val_data_loader, test_data_loader = data_loaders(df_train, df_val, df_test, tokenizer)
    EPOCHS, optimizer, scheduler = setting_hyperparameters(model, len(train_data_loader))
    history = defaultdict(list)
    best_accuracy = 0

    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)

        train_acc, train_prec, train_rec, train_f1, train_prec_micro, train_rec_micro, train_f1_micro, train_loss = train_epoch(
            model,
            train_data_loader,
            optimizer,
            device,
            scheduler,
            len(train_data_loader)
        )

        print(f'Train loss {train_loss} Train accuracy {train_acc}')
        print(f'Train prec macro {train_prec} Train recall macro {train_rec}')
        print(f'Train f1 score macro {train_f1}')
        print(f'Train prec micro {train_prec_micro} Train recall micro {train_rec_micro}')
        print(f'Train f1 score  micro{train_f1_micro}')


        val_acc, val_prec, val_rec, val_f1, val_prec_micro, val_rec_micro, val_f1_micro, val_loss = eval_model(
            model,
            val_data_loader,
            device,
            len(val_data_loader)
        )

        print(f'Val loss {val_loss} Val accuracy {val_acc}')
        print(f'Val prec macro {val_prec} Val recall macro {val_rec}')
        print(f'Val f1 score macro {val_f1}')

        print(f'Val prec micro {val_prec_micro} Val recall micro {val_rec_micro}')
        print(f'Val f1 score  micro {val_f1_micro}')


        print()

        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['train_prec_macro'].append(train_prec)
        history['train_rec_macro'].append(train_rec)
        history['train_f1_macro'].append(train_f1)
        history['train_prec_micro'].append(train_prec_micro)
        history['train_rec_micro'].append(train_rec_micro)
        history['train_f1_micro'].append(train_f1_micro)

        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)
        history['val_prec_macro'].append(val_prec)
        history['val_rec_macro'].append(val_rec)
        history['val_f1_macro'].append(val_f1)
        history['val_prec_micro'].append(val_prec_micro)
        history['val_rec_micro'].append(val_rec_micro)
        history['val_f1_micro'].append(val_f1_micro)

        history_file= os.path.join('output', output_dir,  'history.json')
        with open(history_file, 'w') as writer:
            json.dump(history, writer)


        if val_acc > best_accuracy:
            torch.save(model.state_dict(), os.path.join('output',output_dir, 'xlnet_model_{}.bin'.format(str(epoch))))
            best_accuracy = val_acc
            print('best accuracy: ', best_accuracy )


if __name__ == '__main__':
    import plac
    plac.call(train)
