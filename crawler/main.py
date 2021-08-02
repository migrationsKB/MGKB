import os
import gzip
from itertools import chain
from datetime import datetime
import json
from glob import glob
from collections import OrderedDict

import requests
import numpy as np
import pandas as pd

from utils.api_authen import load_academic_research_brearer
from utils.country_info import CountryInfo
from utils.utils import load_keywords_for_lang


def get_params(cwd):
    ### loading the parameters for downloading the tweets
    with open(os.path.join(cwd, 'src', 'config', 'tweets_fields.json')) as file:
        tweets_fields = json.load(file)

    with open(os.path.join(cwd, 'src', 'config', 'poll_fields.json')) as file:
        poll_fields = json.load(file)

    with open(os.path.join(cwd, 'src', 'config', 'media_fields.json')) as file:
        media_fields = json.load(file)

    with open(os.path.join(cwd, 'src', 'config', 'user_fields.json')) as file:
        user_fields = json.load(file)

    with open(os.path.join(cwd, 'src', 'config', 'place_fields.json')) as file:
        place_fields = json.load(file)

    with open(os.path.join(cwd, 'src', 'config', 'expansions.json')) as file:
        expansions = json.load(file)

    tweets_fields = ','.join(tweets_fields)
    print(tweets_fields)

    poll_fields = ','.join(poll_fields)
    print(poll_fields)

    media_fields = ','.join(media_fields)
    print(media_fields)

    user_fields = ','.join(user_fields)
    print(user_fields)

    place_fields = ','.join(place_fields)
    print(place_fields)

    tweets_expansions = ','.join(expansions)
    print(tweets_expansions)

    return tweets_fields, poll_fields, media_fields, user_fields, place_fields, tweets_expansions


def get_last_start_time(dir):
    # get the last earliest time crawled for tweets
    # as the end time for next crawling.
    files = glob(dir + '/**.gz')
    print('nr of existing files:', len(files))
    if files is not None and files != []:
        dir_dict = {int(filepath.split('_')[1]): filepath for filepath in files}
        od = OrderedDict(sorted(dir_dict.items(), reverse=True))
        first_key = list(od)[0]
        start_time = od[first_key].split('_')[2].replace('.gz','')
        print('last start time:', start_time)
        # year = od[first_key].split('_')[-1].replace('.gz', '')
        # print(year)
        return start_time
    else:
        return None


def query_main(api_name, country_iso2, lang, start_year, end_year):
    cwd = os.getcwd()
    print('current working directory: ', cwd)
    brear_token = load_academic_research_brearer(cwd, api_name)

    # endpoint for academic research
    search_url = 'https://api.twitter.com/2/tweets/search/all'

    tweets_fields, poll_fields, media_fields, user_fields, place_fields, tweets_expansions = get_params(cwd)

    # # already speicify keywords, no lang in query.
    keywords = load_keywords_for_lang(cwd, lang)
    print(keywords)

    # keywords = [
    #     "#refugeesnotwelcome",
    #     "#DeportallMuslims",
    #     "#banislam",
    #     "#banmuslims",
    #     "#destroyislam",
    #     "#norefugees",
    #     "#nomuslims",
    #     "muslim",
    #     "islam",
    #     "islamic",
    #     "immigration",
    #     "migrant",
    #     "refugee",
    #     "asylum"
    # ]
    startdate = start_year + '-01-01T00:00:00.00Z'
    # change end time accordingly.
    enddate = end_year + '-01-01T00:00:00.00Z'
    # set up start_time and end_time parameters in API call
    # max_results, to maximum 500

    # check if the data dir for a country exists.
    output_dir_ = os.path.join(cwd, 'data', 'data', country_iso2)
    if not os.path.exists(output_dir_):
        os.mkdir(output_dir_)

    output_dir = os.path.join(output_dir_, lang)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    query = "({}) has:geo place_country:{} -is:retweet -is:nullcast".format(
        ' OR '.join(keywords), country_iso2)
    print(query)
    print(len(query))
    assert len(query) <= 1024

    ### check the last min id from previous crawling.
    start_time = get_last_start_time(output_dir)
    if start_time is not None:
        print('the min time from last crawling: ', start_time)

        # https://developer.twitter.com/en/docs/twitter-api/tweets/search/api-reference/get-tweets-search-all
        query_params = {'query': query,
                        'tweet.fields': tweets_fields,
                        'user.fields': user_fields,
                        'media.fields': media_fields,
                        'poll.fields': poll_fields,
                        'place.fields': place_fields,
                        'expansions': tweets_expansions,
                        'start_time': startdate, 'end_time': start_time, 'max_results': 500}
    else:
        query_params = {'query': query,
                        'tweet.fields': tweets_fields,
                        'user.fields': user_fields,
                        'media.fields': media_fields,
                        'poll.fields': poll_fields,
                        'place.fields': place_fields,
                        'expansions': tweets_expansions,
                        'start_time': startdate, 'end_time': enddate, 'max_results': 500}

    headers = {"Authorization": "Bearer {}".format(brear_token)}

    ###################query################################
    # connect to end point.
    response = requests.request('GET', search_url, headers=headers, params=query_params)
    print(response.status_code)
    return response, output_dir, country_iso2


def main(api_name, country_iso2, lang, start_year, end_year, flag=True):
    t = datetime.today().strftime('%Y%m%d%H%M%S')
    response, output_dir, country_iso2 = query_main(api_name, country_iso2, lang, start_year,
                                                    end_year)
    while flag:
        if response.status_code == 200:

            #### data #######
            data = response.json()

            data_json = json.dumps(data) + '\n'
            data_encoded = data_json.encode('utf-8')

            ###########################
            df = pd.DataFrame(data['data'])
            dates = df['created_at']
            min_time = np.min(dates)
            print('crawled {} tweets'.format(len(df)))

            # outputfile path.
            outputfile = os.path.join(output_dir, country_iso2 + '_' + t + '_' + str(min_time) + '.gz')

            with gzip.open(outputfile, 'w') as outputfile:
                print('writing tweets to ', outputfile, '....')
                outputfile.write(data_encoded)
            if flag:
                main(api_name, country_iso2, lang, start_year, end_year, flag=True)
            print('*' * 100)

        else:
            flag = False
            raise Exception(response.status_code, response.text)


if __name__ == '__main__':
    # dir = 'data/data/FR'
    # start_time = get_last_start_time(dir)
    # print(start_time)
    import plac
    plac.call(main)

    # main('itflowsapi', 'GB', 'en', '2013', '2014')
