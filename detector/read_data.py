import glob
import json
import gzip
import pandas as pd


def read_dir(directory):
    bots = []
    humans = []
    folder = ['/bots', '/humans']
    name = '/*.json.gz'
    for f in folder:
        paths = glob.glob(directory + f + name)
        for p in paths:
            with gzip.open(p, 'r') as file:
                for line in file:
                    if f == folder[0]:
                        js = json.loads(line)
                        if 'tweets' in js:
                            bots.append(js)
                    elif f == folder[1]:
                        js = json.loads(line)
                        if 'tweets' in js:
                            humans.append(js)

    df_bots = pd.DataFrame(bots)[['screen_name', 'name', 'tweets', 'listed_count',
                                  'followers_count', 'friends_count', 'default_profile_image',
                                  'default_profile', 'statuses_count', 'verified']]
    df_bots['label'] = 'bot'

    df_humans = pd.DataFrame(humans)[['screen_name', 'name', 'tweets', 'listed_count',
                                      'followers_count', 'friends_count', 'default_profile_image',
                                      'default_profile', 'statuses_count', 'verified']]
    df_humans['label'] = 'human'
    frames = [df_bots, df_humans]
    df = pd.concat(frames)

    users = bots + humans
    tweets_texts = []
    num_of_tweets = []
    for u in users:
        tweets = u['tweets']  # a list of dicts
        num_of_tweets.append(len(tweets))
        texts = [t['full_text'] for t in tweets]
        tweets_texts.append(str(texts).strip('[]'))
    df['tweets_texts'] = tweets_texts
    df['num_tweets'] = num_of_tweets
    return df
