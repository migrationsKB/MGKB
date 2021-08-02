import os

import yaml
import tweepy
from tweepy import OAuthHandler


def load_tweepy_api(input_dir, api_name):
    # loading credentials
    credentials = yaml.load(open(os.path.join(input_dir, 'crawler', 'config', 'credentials.yaml')), yaml.FullLoader)
    consumer_key = credentials[api_name]['key']
    consumer_secret = credentials[api_name]['secret']
    access_token = credentials[api_name]['oauth_token']
    access_token_secret = credentials[api_name]['oauth_token_secret']

    # authentication handler
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    # load api via tweepy
    api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
    return api


def load_academic_research_brearer(input_dir, api_name):
    credentials = yaml.load(open(os.path.join(input_dir,'crawler', 'config', 'credentials.yaml')), yaml.FullLoader)
    brearer = credentials[api_name]['bearer_token']
    return brearer