import os
import yaml
import json
from glob import glob
from itertools import islice


def load_keywords_for_lang(input_dir, lang):
    keywords_dict = yaml.load(open(os.path.join(input_dir,'crawler', 'config', 'lang_keywords.yaml')), yaml.FullLoader)
    return keywords_dict[lang]


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def chunks_dictionary(data, SIZE=100):
    it = iter(data)
    for i in range(0, len(data), SIZE):
        yield {k: data[k] for k in islice(it, SIZE)}


def get_keywords_by_category(category_dir='data/extracted/1st_round/hashtags_categories'):
    """
    Get the keywords by category
    :param category:
    :return:
    """
    keywords = []
    # go through hashtags categories load the keywords.
    for file in glob(category_dir + '/**.txt'):
        # print(file)
        with open(file) as reader:
            for line in reader.readlines():
                line = '#' + line.replace('\n', '')
                keywords.append(line)
    return list(set(keywords))

def get_params(cwd):
    """
    Loading the parameters for quering the tweets using API.
    :param cwd: current working directory
    :return:
    """
    config_dir = os.path.join(cwd,'crawler', 'config', 'fields_expansions')

    with open(os.path.join(config_dir, 'tweets_fields.json')) as file:
        tweets_fields = json.load(file)

    with open(os.path.join(config_dir, 'poll_fields.json')) as file:
        poll_fields = json.load(file)

    with open(os.path.join(config_dir, 'media_fields.json')) as file:
        media_fields = json.load(file)

    with open(os.path.join(config_dir, 'user_fields.json')) as file:
        user_fields = json.load(file)

    with open(os.path.join(config_dir, 'place_fields.json')) as file:
        place_fields = json.load(file)

    with open(os.path.join(config_dir, 'expansions.json')) as file:
        expansions = json.load(file)

    tweets_fields = ','.join(tweets_fields)

    poll_fields = ','.join(poll_fields)

    media_fields = ','.join(media_fields)

    user_fields = ','.join(user_fields)
    # print(user_fields)

    place_fields = ','.join(place_fields)
    # print(place_fields)

    tweets_expansions = ','.join(expansions)
    # print(tweets_expansions)

    return tweets_fields, poll_fields, media_fields, user_fields, place_fields, tweets_expansions
