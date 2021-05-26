import os
import re
import gzip
import string
from glob import glob

from spacy.lang.en.stop_words import STOP_WORDS as stopwords
from joblib import Parallel, delayed
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
import preprocessor as p
from pandarallel import pandarallel
from unidecode import unidecode

from preprocessing.preprocessing_sentence import expandContractions

############### define options for tweets####################################
###### delete also url############################################
p.set_options(p.OPT.MENTION, p.OPT.RESERVED, p.OPT.SMILEY, p.OPT.EMOJI, p.OPT.NUMBER, p.OPT.URL)

###########customize stopwords#########################################
# reserved_stopwords = ['no', 'nor', 'not', 'needn', 'aren', 'ain', 'couldn', 'didn',
#                       'hadn', 'hasn', 'haven', 'isn', 'mightn', 'mustn',
#                       'shan', 'shan', 'shouldn']

reserved_stopwords = ['cannot', 'nor', 'nobody', 'not', 'no', 'nothing', 'none', 'noone', 'nowhere', 'never', 'needn', 'aren', 'ain', 'couldn', 'didn', 'hadn', 'hasn', 'haven', 'isn', 'mightn', 'mustn', 'shan', 'shan', 'shouldn']

stopwords = [word for word in stopwords if word not in reserved_stopwords]
#######################################################################
special_chars = ['&nbsp;', '&lt;', '&gt;', '&amp;', '&quot;', '&apos;',
                 '&cent;', '&pound;', '&yen;', '&euro;', '&copy;', '&reg;',
                 '£']

#### initialize panda parallel###########
pandarallel.initialize()

punctuations = '!"$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'

wordnet_lemmatizer = WordNetLemmatizer()
porter = PorterStemmer()


###########CLEAN TEXT FOR TOPIC MODELING#######################
def clean_text(text):
    """for topic modeling"""
    text = text.replace('\n', ' ').replace('\r', ' ').lower()
    # expand contractions
    contraction_expanded = expandContractions(text)
    # remove urls, mentions, smilelys, etc.
    cleaned_text = p.clean(contraction_expanded)

    # replace speical chars with space
    for ent in special_chars:
        cleaned_text = cleaned_text.replace(ent, " ")
    # unidecode a text
    unidecoded = unidecode(cleaned_text)

    # remove punctuations except '#'
    punct_removed = unidecoded.translate(str.maketrans(' ', ' ', punctuations))

    # remove stopwords
    tokens = punct_removed.split()
    ### not stemming hashtags
    tokens_ = [wordnet_lemmatizer.lemmatize(token) if not token.startswith('#') and '_' not in token else token for token in tokens]
    tokens_removed_stopwords = [token.replace('#','') for token in tokens_ if token not in stopwords]
    tokens = [token for token in tokens_removed_stopwords if len(token)>2]
    return tokens


##### clean text for sentiment analysis##############################
def contains_punctuation(w):
    return any(char in punctuations for char in w)


def contains_numeric(w):
    return any(char.isdigit() for char in w)


def clean_s(doc):
    doc = doc.replace('\n','').replace('\r','')
    
    # remove urls, mentions, smilelys, etc.
    doc = p.clean(doc)
    if '\\u' in doc:
        doc = doc.encode().decode("unicode-escape")
        
    doc = doc.lower()
    cleaned_text = expandContractions(doc)

    # replace speical chars with space
    for ent in special_chars:
        cleaned_text = cleaned_text.replace(ent, " ")
    # unidecode a text
    # unidecoded = unidecode(cleaned_text)
    punct_removed = cleaned_text.translate(str.maketrans(punctuations, ' ' * len(punctuations))).replace('#', ' #')
#     punct_removed = cleaned_text.translate(str.maketrans(' ', ' ', string.punctuation))

    # remove stopwords
    tokens = punct_removed.split()
    ### not stemming hashtags
    ### lemmatize for sentiment analysis
#     tokens_ = [wordnet_lemmatizer.lemmatize(token) if not token.startswith('#') else token for token in tokens]
    tokens_removed_stopwords = [token.replace('#','') for token in tokens if token not in stopwords]
    tokens_removed_nrs = [i for i in tokens_removed_stopwords if not i.isdigit()]
    sentence = ' '.join(tokens_removed_nrs)
    
    return sentence


if __name__ == '__main__':
    text = "“@intifada: Some 20,000 Palestinian refugees I'm &amp; “ trapped in Syria's Yarmouk camp, denied access " \
           "to food http://t.co/KRCNPMLfin #IamWithRefugee”"
    cleaned_text = clean_text(text)
    print(text)
    print(cleaned_text)
