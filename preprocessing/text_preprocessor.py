import unicodedata
from pathlib import Path
import os
from typing import List

from tqdm import tqdm
import spacy
import pandas as pd
import preprocessor as p
import string

from pandarallel import pandarallel

pandarallel.initialize()

rootdir = Path(__file__).parent
print('rootdir: ', rootdir)

p.set_options(p.OPT.MENTION, p.OPT.RESERVED, p.OPT.SMILEY, p.OPT.EMOJI, p.OPT.NUMBER, p.OPT.URL)

special_chars = ['&nbsp;', '&lt;', '&gt;', '&amp;', '&quot;', '&apos;', '&cent;', '&pound;', '&yen;', '&euro;',
                 '&copy;', '&reg;', '£']

nlp = spacy.load("en_core_web_sm")


def preprocess_one_tweet(text):
    text = p.clean(text)
    text = text.replace('#', ' ')

    for char in special_chars:
        text = text.replace(char, ' ')

    # remove accented char2acters
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

    doc = nlp(text)
    doc = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    doc = [word.lower().translate(str.maketrans('', '', string.punctuation + "0123456789")) for word in doc]
    doc = [w for w in doc if len(w) >= 2]
    doc = " ".join(doc)
    doc = " ".join(doc.split())
    return doc


def text_processing(docs: List, output_dir: str = os.path.join(rootdir, 'preprocessed_data'),
                    outputfile='processed_tweets') -> List:
    """
    Preprocessing the texts: remove punctuations, remove the texts whose length are less than 2
    """
    ### clean tweets
    docs = [preprocess_one_tweet(doc) for doc in tqdm(docs)]

    print('size of docs:', len(docs))
    ### save the raw text.
    with open(os.path.join(output_dir, outputfile), 'w') as writer:
        for line in docs:
            writer.write(line + '\n')


if __name__ == '__main__':
    df = pd.read_csv('data/preprocessed/df_geo.csv', index_col=0)
    df['preprocessed_text'] = df['text'].parallel_apply(preprocess_one_tweet)  # 384891

    df_dropna = df.dropna(subset=['preprocessed_text'])  # 384752

    df_dedup = df_dropna.drop_duplicates(subset=['preprocessed_text'])  # 360377
    df_sorted = df_dedup.sort_values(by='created_at')
    df_sorted.to_csv('data/preprocessed/df_geo_text_preprocessed.csv')