from itertools import chain

import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn import metrics

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

def accuracy(pred, label):
    prob, idx = torch.max(pred,1)
    precision = (idx==label).float().mean()
    if gpu:
        accuracy.append(precision.data.item())
    else:
        accuracy.append(precision.data.numpy()[0])
    return np.mean(accuracy)

def cat_accuracy(pred, label):
    max_preds = pred.argmax(dim=1, keepdim=True)
    correct = max_preds.squeeze(1).eq(label)
    #correct = torch.LongTensor([0,4,0,0]).to(device).eq(label)
    correct = correct.sum().unsqueeze(0)
    bs = torch.LongTensor([label.shape[0]]).to(device)
    acc = correct.item() / bs.item()
    #return correct.sum()/torch.LongTensor([label.shape[0]])
    return acc


def train_model(model, it, lossf, optimizer):
    model.train()
    ep_loss = 0.0
    ep_acc = 0.0
    
    targets, predictions = [], []
    for b in tqdm(it):
        optimizer.zero_grad()
        seq, label = b.text, b.label
        pred = model(seq)
        loss = lossf(pred, label)
        
        pred = pred.argmax(dim=1, keepdim=True)
        loss.backward()
        optimizer.step()
        ep_loss += loss.item()
        
        
        label = label.cpu().detach().numpy()
        pred = pred.cpu().detach().numpy()
        acc = metrics.accuracy_score(label, pred)
        ep_acc += acc
        
        targets.append(label)
        predictions.append(label)
        
    prediction_list = list(chain.from_iterable([list(x) for x in predictions]))
    target_list = list(chain.from_iterable([list(x) for x in targets]))
    return ep_loss/ len(it), ep_acc/ len(it), target_list, prediction_list

def evaluate_model(model, it, lossf):
    model.eval()
    ep_loss = 0.0
    ep_acc = 0.0
        
    targets, predictions = [], []

    with torch.no_grad():
        for b in it:
            seq, label = b.text, b.label
            pred = model(seq)
            loss = lossf(pred, label)
            
            pred = pred.argmax(dim=1, keepdim=True)
            
            ep_loss += loss.item()
            
            label = label.cpu().detach().numpy()
            pred = pred.cpu().detach().numpy()
            acc = metrics.accuracy_score(label, pred)
            ep_acc += acc
            
            targets.append(label)
            predictions.append(pred)
            
    prediction_list = list(chain.from_iterable([list(x) for x in predictions]))
    target_list = list(chain.from_iterable([list(x) for x in targets]))        
    
    return ep_loss/ len(it), ep_acc/ len(it),target_list, prediction_list




# Taken from the scikit-learn documentation
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def evaluate_test(targets, preds):
    #### convert prediction matrix to prediction labels
    accuracy = metrics.accuracy_score(targets, preds)
    
    precision = metrics.precision_score(targets, preds,labels=np.unique(preds),average='macro')
    recall = metrics.recall_score(targets, preds, labels=np.unique(preds), average='macro')
    f1_score = metrics.f1_score(targets, preds,labels=np.unique(preds), average='macro')

    precision_micro = metrics.precision_score(targets, preds, labels=np.unique(preds), average='micro')
    recall_micro = metrics.recall_score(targets, preds, labels=np.unique(preds), average='micro')
    f1_score_micro = metrics.f1_score(targets, preds, labels=np.unique(preds), average='micro')
    
    print('accruacy: ', accuracy)
    print('precision macro: ', precision)
    print('recall macro:', recall)
    print('f1 score macro: ', f1_score)
    
    print('precision micro: ', precision_micro)
    print('recall micro: ', recall_micro)
    print('f1 score micro: ', f1_score_micro)
    
    
    
    
