import os
import gzip
from datetime import datetime
import time
import json
from glob import glob
from collections import OrderedDict

import requests
import numpy as np
import pandas as pd

from utils.api_authen import load_academic_research_brearer
from utils.utils import get_params, load_keywords_for_lang


def get_last_start_time(dir):
    """
    Get the last earliest crawled dates for tweets as the end time for next crawling.
    :param dir: Directory for storing the tweets.
    :return:
    """
    files = glob(dir + '/**.gz')
    print('nr of existing files:', len(files))
    if files is not None and files != []:
        print(files)
        dir_dict = {int(filepath.split('_')[1]): filepath for filepath in files}
        od = OrderedDict(sorted(dir_dict.items(), reverse=True))
        first_key = list(od)[0]
        start_time = od[first_key].split('_')[2].replace('.gz', '')
        print('last start time:', start_time)
        return start_time
    else:
        return None


def query_main(api_name, country_iso2, keywords, idx, lang, start_year, end_year):
    """
    specify hashtag operations.
    :param api_name:
    :param country_iso2:
    :param keywords:
    :param lang:
    :param start_year:
    :param end_year:
    :return:
    """
    #### crawling with hashtags categories from first round
    cwd = os.getcwd()
    print('current working directory: ', cwd)
    brear_token = load_academic_research_brearer(cwd, api_name)

    # endpoint for academic research
    search_url = 'https://api.twitter.com/2/tweets/search/all'

    tweets_fields, poll_fields, media_fields, user_fields, place_fields, tweets_expansions = get_params(cwd)


    startdate = start_year + '-01-01T00:00:00.00Z'
    # change end time accordingly.
    enddate = end_year + '-08-02T00:00:00.00Z'
    # set up start_time and end_time parameters in API call
    # max_results, to maximum 500

    # check if the data dir for a country exists.
    # idx batch of keywords
    keywords_ = keywords[idx]
    output_dir_root = os.path.join(cwd, 'data', '3rd-round-data', country_iso2)
    if not os.path.exists(output_dir_root):
        os.mkdir(output_dir_root)

    output_dir_ = os.path.join(output_dir_root, str(idx))
    if not os.path.exists(output_dir_):
        os.mkdir(output_dir_)

    output_dir = os.path.join(output_dir_, lang)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)


    query = "({}) has:geo place_country:{} -is:retweet -is:nullcast".format(
        ' OR '.join(keywords_), country_iso2)
    print(query)
    print('query length => ', len(query))
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
                        'start_time': startdate, 'end_time': start_time, 'max_results': 100}
    else:
        query_params = {'query': query,
                        'tweet.fields': tweets_fields,
                        'user.fields': user_fields,
                        'media.fields': media_fields,
                        'poll.fields': poll_fields,
                        'place.fields': place_fields,
                        'expansions': tweets_expansions,
                        'start_time': startdate, 'end_time': enddate, 'max_results': 100}

    # https://github.com/twitterdev/Twitter-API-v2-sample-code/blob/main/Full-Archive-Search/full-archive-search.py
    headers = {"Authorization": "Bearer {}".format(brear_token),
               "User-Agent": "v2FullArchiveSearchPython" }

    ###################query################################
    # connect to end point.
    response = requests.request('GET', search_url, headers=headers, params=query_params)
    print(response.status_code)
    return response, output_dir, country_iso2


def main(api_name, country_iso2, keywords, idx, lang, start_year, end_year, flag=True):
    t = datetime.today().strftime('%Y%m%d%H%M%S')
    response, output_dir, country_iso2 = query_main(api_name, country_iso2, keywords, idx, lang, start_year,
                                                    end_year)
    while flag:

        if response.status_code == 200:

            #### data #######
            data = response.json()

            data_json = json.dumps(data) + '\n'
            data_encoded = data_json.encode('utf-8')

            ###########################
            try:
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
                    main(api_name, country_iso2, keywords, idx, lang, start_year, end_year, flag=True)
                print('*' * 100)
            except Exception:
                if idx < 58:
                    idx += 1
                    print('idx:', idx)
                    main(api_name, country_iso2, keywords, idx, lang, start_year, end_year, flag=True)
                else:
                    break

        else:
            # response code ==429, rate limit exceeded. 100 api calls finished, wait another hour.

            flag = False
            raise Exception(response.status_code, response.text)


if __name__ == '__main__':
    # import plac
    # plac.call(call_main)
    destination_countries = ['GB']
    # finished :PL, GB, NL, DE, CH, AT, ES, FR, HU,IT, SE
    # for country in destination_countries:
    # keywords_list = get_keywords_by_category()
    # keywords_chunks = list(chunks(keywords_list, 40))
    with open('data/extracted/keywords_chunked_all.json') as file:
        keywords_chunks = json.load(file)
    # idx =
    # main('itflowsapi', 'GB', keywords_chunks, idx, 'en', '2021', '2022')
    for x in range(42, 58):
        main('itflowsapi', 'GB', keywords_chunks, x, 'en', '2021', '2021')
