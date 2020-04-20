import click
import sys
import pickle

import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import scale, StandardScaler
from scipy.sparse import hstack  # "horizontal stack"
from . import credentials_path, clf_path
from .read_data import read_dir

# @click.group()
# def main(args=None):
#     """Console script for detector."""
#     return 0


def make_features(df):
    ## Add your code to create features.
    vec = DictVectorizer()
    feature_dicts = []
    for i, row in df.iterrows():
        tweets = row['tweets']
        texts = [t['full_text'] for t in tweets]
        features = get_user_features(texts, [row.tweets_texts],
                                     row.num_tweets,
                                     row.followers_count,
                                     row.listed_count,
                                     row.friends_count,
                                     row.default_profile_image,
                                     row.default_profile,
                                     row.statuses_count,
                                     row.verified,
                                     row.name,
                                     row.screen_name)
        feature_dicts.append(features)
    X = vec.fit_transform(feature_dicts)
    print(X.shape)
    return X, vec

def levenshtein(seq1, seq2):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros ((size_x, size_y))
    for x in range(size_x):
        matrix [x, 0] = x
    for y in range(size_y):
        matrix [0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix [x,y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix [x,y] = min(
                    matrix[x-1,y] + 1,
                    matrix[x-1,y-1] + 1,
                    matrix[x,y-1] + 1
                )
#     print (matrix)
    return (matrix[size_x - 1, size_y - 1])


def get_user_features(texts, tweets_texts, num_of_tweets, followers_count, listed_count, friends_count,
                      default_profile_image, default_profile, statuses_count, verified, name, screen_name):
    count_mention = 0
    count_url = 0
    factor = 100
    features = {}  # a dict

    # count http and mentions
    for s in texts:
        if 'http' in s:
            count_url += 1
        if '@' in s:
            count_mention += 1

    if len(texts) == 0:
        features['tweets_avg_urls'] = 0
        features['tweets_avg_mentions'] = 0
    else:
        features['tweets_avg_urls'] = factor * count_url / len(texts)
        features['tweets_avg_mentions'] = factor * count_mention / len(texts)

    features['followers_count'] = followers_count
    features['listed_count'] = listed_count
    features['friends_count'] = friends_count
    features['default_profile_image'] = int(default_profile_image)
    features['default_profile'] = int(default_profile)
    features['verified'] = int(verified)
    features['statuses_count'] = statuses_count

    # add the diff_user_screen_name feature
    levenshtein_distance = levenshtein(str(name), str(screen_name))
    features['distance_between_names'] = levenshtein_distance

    # add the ratio_followers feature
    if followers_count + friends_count == 0:
        features['ratio_followers'] = 0
    else:
        features['ratio_followers'] = factor * followers_count / (followers_count + friends_count)
    # add the tri_gram feature
    tri_count_vec = CountVectorizer(min_df=1, max_df=1.0, ngram_range=(2, 3))
    try:
        user_words = tri_count_vec.fit_transform(tweets_texts)  # 这个方法返回的是什么？
    except ValueError:
        features['tri_gram_most_common'] = 0
        return features

    freqs = zip(tri_count_vec.get_feature_names(), user_words.sum(axis=0).tolist()[0])
    # sort from largest to smallest
    f_list = sorted(freqs, key=lambda x: -x[1])
    top_element = f_list[0:3]  # a zip of word, freq
    #     top_word = top_element[0]
    #     top_freq = top_element[1]
    #     print(top_element)

    freq_sum = 0
    for word, freq in top_element:
        freq_sum += freq
    if num_of_tweets != 0:
        #         frequency = top_freq / num_of_tweets * 100
        frequency = freq_sum / num_of_tweets * 100
    else:
        frequency = 0
    features['tri_gram_most_common'] = frequency
    return features


# @main.command('hello')
# def hello():
#     print("hello world")

# @main.command('train')
# @click.argument('directory', type=click.Path(exists=True))
def train(directory):
    """
    Train a classifier and save it.
    """
    print('reading from %s' % directory)
    # (1) Read the data...
    #
    df = read_dir(directory)
    print(df.label.value_counts())

    print("start making features...")
    # (2) Create classifier and vectorizer.
    X, dict_vec = make_features(df)
    print(dict_vec.get_feature_names())
    print("finished making features.")

    min_df = 0.05
    max_df = 0.50
    print("min_df=%.2f, max_df=%.2f"%(min_df, max_df))
    count_vec = CountVectorizer(min_df=min_df, max_df=max_df, ngram_range=(3, 3))

    X_words = count_vec.fit_transform(df.tweets_texts)
    print(X_words.shape)
    optimal_X_all = hstack([X, X_words]).tocsr()
    scaler = StandardScaler(with_mean=False)  # optionally with_mean=False to save memory (keep matrix sparse)
    optimal_X_all = scaler.fit_transform(optimal_X_all)
    # optimal_X_all = scaler.fit_transform(optimal_X_all.todense())

    print("finished optimal_X_all.")

    clf = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=10000, C=1, penalty='l2')
    # print("Using MLP.")
    # clf = MLPClassifier(hidden_layer_sizes=10, activation='relu', solver='adam', alpha=0.1, max_iter=10000)
    # clf = MLPClassifier(hidden_layer_sizes=5000, activation='relu', solver='adam', alpha=0.001, max_iter=100000)
    # rand = RandomForestClassifier(n_estimators=300, min_samples_leaf=1)
    y = np.array(df.label)
    ## no reason to .fit here since you do it after cross validation. -awc
    # clf.fit(optimal_X_all, y)
    # print("finished clf fit.")

    # (3) do cross-validation and print out validation metrics
    # (classification_report)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []
    truths = []
    predictions = []
    for train, test in kf.split(optimal_X_all):
        clf.fit(optimal_X_all[train], y[train])
        pred = clf.predict(optimal_X_all[test])
        accuracies.append(accuracy_score(y[test], pred))
        truths.extend(y[test])
        predictions.extend(pred)
    print('accuracy over all cross-validation folds: %s' % str(accuracies))
    print('mean=%.2f std=%.2f' % (np.mean(accuracies), np.std(accuracies)))
    print("classification_report: \n", classification_report(truths, predictions))

    # (4) Finally, train on ALL data one final time and
    # train...
    # save the classifier
    clf.fit(optimal_X_all, y)
    print_top_features(dict_vec, count_vec, clf)
    pickle.dump((clf, count_vec, dict_vec, scaler), open(clf_path, 'wb'))

def print_top_features(vec, count_vec, clf):
    # sort coefficients by class.
    features = vec.get_feature_names() + count_vec.get_feature_names()
    coef = [-clf.coef_[0], clf.coef_[0]]
    for ci, class_name in enumerate(clf.classes_):
        print('top 15 features for class %s:' % class_name)
        for fi in coef[ci].argsort()[-1:-16:-1]:  # descending order.
        # for fi in coef[ci].argsort()[-1:-56:-1]:  # descending order.
            print('%20s\t%.6f' % (features[fi], coef[ci][fi]))
        print()

# if __name__ == "__main__":
#     sys.exit(main())  # pragma: no cover
