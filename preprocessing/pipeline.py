import os
import re
import gzip
import string
from glob import glob

import inflect
from nltk.corpus import words
from ekphrasis.classes.segmenter import Segmenter
from ekphrasis.classes.spellcorrect import SpellCorrector
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacymoji import Emoji
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER
from spacy.lang.char_classes import CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_infix_regex
from spacy.lang.en.stop_words import STOP_WORDS as stopwords
from joblib import Parallel, delayed
import preprocessor as p
from pandarallel import pandarallel

import pandas as pd
import numpy as np

from src.preprocessing.preprocessing_sentence import expandContractions
from src.preprocessing.preprocessing_sentence import prevent_sentence_boundaries
from src.preprocessing.preprocessing_word import remove_repeated_characters

from src.utils.reader import read_gz_file

######################
words_dict= words.words()

###inflect engine#####
inflect_p = inflect.engine()

### modify infixer => do not split up the words with hyphens.
### https://spacy.io/usage/linguistic-features#native-tokenizer-additions###

CONCAT_QUOTES = CONCAT_QUOTES.replace('\'', '')
infixes = (
        LIST_ELLIPSES
        + LIST_ICONS
        + [
            r"(?<=[0-9])[+\-\*^](?=[0-9-])",
            r"(?<=[{al}{q}])\.(?=[{au}{q}])".format(
                al=ALPHA_LOWER, au=ALPHA_UPPER, q=CONCAT_QUOTES
            ),
            r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA),
            # ✅ Commented out regex that splits on hyphens between letters:
            # r"(?<=[{a}])(?:{h})(?=[{a}])".format(a=ALPHA, h=HYPHENS),
            r"(?<=[{a}0-9])[:<>=/](?=[{a}])".format(a=ALPHA),
        ]
)

infix_re = compile_infix_regex(infixes)

segmenter = Segmenter(corpus='twitter')
# corrector = SpellCorrector(corpus='english')

#############load initialize nlp###################################
nlp = spacy.load('en_core_web_sm', disable=['tagger', 'parser', 'ner'])
nlp.tokenizer.infix_finditer = infix_re.finditer
emoji = Emoji(nlp, merge_spans=False)
nlp.add_pipe(prevent_sentence_boundaries)
nlp.add_pipe(emoji)

############### define options for tweets####################################
### delete also url############################################
p.set_options(p.OPT.MENTION, p.OPT.RESERVED, p.OPT.SMILEY, p.OPT.EMOJI, p.OPT.NUMBER, p.OPT.URL)

###########customize stopwords#########################################
reserved_stopwords = ['no', 'nor', 'not', 'needn', 'aren', 'ain', 'couldn', 'didn',
                      'hadn', 'hasn', 'haven', 'isn', 'mightn', 'mustn',
                      'shan', 'shan', 'shouldn']

STOP_WORDS = [word for word in stopwords if word not in reserved_stopwords]
#######################################################################
special_chars = ['&nbsp;', '&lt;', '&gt;', '&amp;', '&quot;', '&apos;',
                 '&cent;', '&pound;', '&yen;', '&euro;', '&copy;', '&reg;',
                 '£']

### initialize panda parallelization#############
pandarallel.initialize()
    
#############################################################
def cleaner_doc(doc, entities, hashtags):
    "Extract relevant text from DataFrame using a regex"
    pattern = r"[A-Za-z0-9\-]{1,50}"
    # delete special chars
    for ent in special_chars:
        doc = doc.replace(ent, ' ')

    ################################################
    # TODO: try to filter out entities first
    ## WITH FREQUENCY, OR SOMETHING ELSE.
    to_be_ignored = []
    if entities != []:
        for entity in entities:
            if len(entity.split()) > 0:
                entity_ = '-'.join(entity.split())
                to_be_ignored.append(entity_)
                doc = doc.replace(entity, entity_)
            else:
                to_be_ignored.append(entity)

    ##### CLEAN HASHTAGS.############################
    ### SEGMENT AND CORRECT#######################
    ##### HOW TO DO IT FAST!#######################
    if hashtags != []:
        for hashtag in hashtags:
            hashtag_ = segmenter.segment(hashtag)
            if len(hashtag_.split()) > 0:
                hashtag_ = '-'.join(hashtag_.split())
                doc = doc.replace(hashtag, hashtag_)
                to_be_ignored.append(hashtag_)
            else:
                to_be_ignored.append(hashtag)

    # expand contractions
    contraction_expanded = expandContractions(doc)
    # preprocess tweets, remove hashtags, mentions, urls, smileys, reserved
    processed_tweet = p.clean(contraction_expanded)
    # Regex pattern for only alphanumeric, hyphenated text with 3 or more chars
    processed_text = re.findall(pattern, processed_tweet)

    ################remove repeated characters.
    text_processed = []
    for tok in processed_text:
        if tok not in set(to_be_ignored):
            #### get singular noun.
            if inflect_p.singular_noun(tok):
                tok_singular = inflect_p.singular_noun(tok)
                text_processed.append(tok_singular)
            else:
                tok_removed_repeated = remove_repeated_characters(tok.lower(), words_dict)
                ### avoid the words like "-abc-"
                if tok.lower()==tok_removed_repeated:
                    tok = tok
                else:
                    tok = tok_removed_repeated
                tok_ = tok.split('-')
                tok_ = [w for w in tok_ if w != '']
                tok = '-'.join(tok_)
                text_processed.append(tok)
        else:
            text_processed.append(tok)

    doc_ = ' '.join(text_processed)
    return doc_


def lemmatize_emoji_pipe(doc):
    doc_ = []
    for token in doc:
        if token._.emoji_desc is not None:
            doc_.append(token._.emoji_desc)
        else:
            if '-' in token.text:
                doc_.append(token.text)
            else:
                if token.text.lower() not in STOP_WORDS:
                    if token.text.isalpha():
                        if not token.is_punct:
                            tok = str(token.lemma_).lower()
                            doc_.append(tok)
    return doc_


def chunker(iterable, total_length, chunksize):
    return (iterable[pos: pos + chunksize] for pos in range(0, total_length, chunksize))


def flatten(list_of_lists):
    "Flatten a list of lists to a combined list"
    print('flatten +1')
    return [item for sublist in list_of_lists for item in sublist]


###################### lemmatize and transform emoji#######################
def process_chunk(texts):
    preproc_pipe = []
    for doc in nlp.pipe(texts, batch_size=20):
        preproc_pipe.append(lemmatize_emoji_pipe(doc))
    return preproc_pipe


def preprocess_parallel(df, chunksize=100):
    executor = Parallel(n_jobs=os.cpu_count(), backend='multiprocessing', prefer="processes")
    do = delayed(process_chunk)
    tasks = (do(chunk) for chunk in chunker(df['clean'], len(df), chunksize=chunksize))
    result = executor(tasks)
    return flatten(result)


def get_df_from_tweet(data):
    ### hashtag, entity,annotations, text, id
    hashtags = []
    entities_annotated = []
    texts = []
    ids = []
    for item in data:
        if 'id' in item:
            ids.append(item['id'])
            text = item['text'].replace('\n','').replace('\r','')
            texts.append(text)

            norm_tags = []
            hash_tags = []
            if 'entities' in item:
                entities = item['entities']
                if 'hashtags' in entities:
                    for hs in entities['hashtags']:
                        hash_tags.append(hs['tag'])

                if 'annotations' in entities:
                    annotations = entities['annotations']
                    for anno in annotations:
                        norm_text = anno['normalized_text'].translate(str.maketrans('', '', string.punctuation)).strip()
                        norm_tags.append(norm_text)

            hashtags.append(hash_tags)
            entities_annotated.append(norm_tags)

    return hashtags, ids, texts, entities_annotated


def main(file_dir):
    """
    the output_file should be with the endung ".csv.gz"
    :param file_dir:
    :param output_file:
    :return:
    """
    df = pd.DataFrame()

    hashtags = []
    entities_annotated = []
    texts = []
    ids = []
    count = 0
    for filepath in glob(file_dir + '/**.gz'):
        print(filepath)
        filedata = read_gz_file(filepath)
        hashtags_, ids_, texts_, entities_annotated_ = get_df_from_tweet(filedata['data'])

        hashtags += hashtags_
        entities_annotated += entities_annotated_
        texts += texts_
        ids += ids_

        count += len(texts_)

    df['hashtags'] = hashtags
    df['id'] = ids
    df['text'] = texts
    df['entities'] = entities_annotated
    #####################################################

    print(count)
    return df


if __name__ == '__main__':
    filepath = 'data/test/GB_20210206021606_2020-08-11T22:09:26.000Z_2020.gz'
    # df = main('data/test', 'data', 'GB_output')
    # print(df.clean.to_list()[:10])
    # print(df.processed_text.to_list()[:10])
    # print(df.text.to_list()[:10])
    # print(df.hashtags.to_list()[:10])
    # print(df.entities.to_list())
    # print(df.id.to_list()[:10])

    test = " abc- abdc-abdc- -abc uk maria"
    entities = ["uk maria"]
    hashtags = []
    doc = cleaner_doc(test, entities, hashtags)
    print(doc)
    # main('data/test', 'data', 'GB_output')
