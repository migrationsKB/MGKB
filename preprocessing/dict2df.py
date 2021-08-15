import json

import pandas as pd
import numpy as np
from pandarallel import pandarallel

pandarallel.initialize()


#################helper functions#####################
def get_long(geo):
    if str(geo) != 'nan':
        if 'coordinates' in geo:
            return geo['coordinates']['coordinates'][0]
        else:
            return None
    else:
        return None


def get_lat(geo):
    if str(geo) != 'nan':
        if 'coordinates' in geo:
            return geo['coordinates']['coordinates'][1]
        else:
            return None
    else:
        return None


def get_place_id(geo):
    if str(geo) != 'nan':
        if 'place_id' in geo:

            return geo['place_id']
        else:
            return None
    else:
        return None


def get_hashtags(entities):
    if str(entities) != 'nan':
        if 'hashtags' in entities:
            return [x['tag'] for x in entities['hashtags']]
        else:
            return None
    else:
        return None


def get_mentions(entities):
    if str(entities) != 'nan':
        if 'mentions' in entities:
            return [x['username'] for x in entities['mentions']]
        else:
            return None
    else:
        return None


def get_retweet(public_metrics):
    return public_metrics['retweet_count']


def get_like(public_metrics):
    return public_metrics['like_count']


def get_reply(public_metrics):
    return public_metrics['reply_count']


##########################################


def dic2df(tweet_dict, country_code):
    tweets = tweet_dict['data']
    places = tweet_dict['places']
    # get geo information
    place_df = pd.DataFrame.from_dict(places, orient='index')
    place_df.rename(columns={'id': 'place_id'}, inplace=True)
    # tweets data df
    df = pd.DataFrame.from_dict(tweets, orient='index')
    # coordinates
    df['long'] = df['geo'].parallel_apply(get_long)
    df['lat'] = df['geo'].parallel_apply(get_lat)
    # prepare to use place_df
    df['place_id'] = df['geo'].parallel_apply(get_place_id)
    # hashtags, user mentions
    df['hashtags'] = df['entities'].parallel_apply(get_hashtags)
    df['user_mentions'] = df['entities'].parallel_apply(get_mentions)
    # user interactions
    df['reply_count'] = df['public_metrics'].parallel_apply(get_reply)
    df['like_count'] = df['public_metrics'].parallel_apply(get_like)
    df['retweet_count'] = df['public_metrics'].parallel_apply(get_retweet)
    # merge df and place_df
    merged = pd.merge(df, place_df, left_on='place_id', right_on='place_id', how='left')
    merged.rename(columns={'geo_y': 'geo'}, inplace=True)
    columns_reserved = ['author_id', 'conversation_id', 'text',
                        'id', 'created_at', 'lang',
                        'long', 'lat',
                        'hashtags', 'user_mentions',
                        'reply_count', 'like_count',
                        'retweet_count',
                        'full_name', 'name', 'country', 'geo',
                        'country_code']
    merged = merged[columns_reserved]
    merged['country_code'] = [country_code for x in range(len(merged))]
    return merged


def main(input_file='data/preprocessed/123-data-restructured.json'):
    with open(input_file) as reader:
        tweet_dict_all = json.load(reader)
    df_ls = []
    for country_code, tweet_dict in tweet_dict_all.items():
        df_country = dic2df(tweet_dict, country_code)
        print(country_code, len(df_country))
        df_ls.append(df_country)

    df= pd.concat(df_ls)
    df.index=df.id
    print(len(df))
    df.to_csv('data/preprocessed/df_geo.csv')

if __name__ == '__main__':
    main()