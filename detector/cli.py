import click
from detector import credentials_path, clf_path
from detector.model import train
from detector.mytwitter import Twitter
import sys
from detector.model import get_user_features
import pickle
import sys
import numpy as np
from detector.app.forms import MyForm
from flask import render_template, flash, redirect, session, request


twapi = Twitter(credentials_path)
clf, count_vec, dict_vec, scaler = pickle.load(open(clf_path, 'rb'))

from flask import Flask

app = Flask(__name__)
app.config['SECRET_KEY'] = 'you-will-never-guess'  # for CSRF

@click.group()
def main(args=None):
    """Console script for osna."""
    return 0

@main.command('web')
@click.option('-t', '--twitter-credentials', required=False, type=click.Path(exists=True), show_default=True,
              default=credentials_path, help='a json file of twitter tokens')
@click.option('-p', '--port', required=False, default=5000, show_default=True, help='port of web server')
def web(twitter_credentials, port):
    # from .app import app
    app.run(host='0.0.0.0', debug=True, port=port)

@main.command('train')
@click.argument('directory', type=click.Path(exists=True))
def train_model(directory):
    train(directory)

@main.command('predict')
@click.argument('directory', type=click.Path(exists=True))
def test(directory):
    f = open(directory)
    Lines = f.readlines()
    total = 0
    count = 0
    for line in Lines:
        name = line.strip()
        tweet_objects = [t for t in twapi.tweets_for_screen_name(name, limit=200)]
        # tweets = [t['full_text'] for t in tweet_objects]
        prediction = ''
        probas = 0
        if len(tweet_objects) != 0:
            X_all, prediction = get_prediction(tweet_objects)
            probas = clf.predict_proba(X_all)
            total += 1
        if prediction == 'bot':
            count += 1
        print("name: {}, prediction: {}".format(name, prediction))
        print('probas=', probas)
    print("accuracy: {:.2f}".format(count / total))

@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    form = MyForm()
    result = None
    if form.validate_on_submit():
        input_field = form.input_field.data
        print(input_field)
        tweet_objects = [t for t in twapi._get_tweets('screen_name', input_field, limit=200)]
        tweets = [t['full_text'] for t in tweet_objects]
        if len(tweet_objects) == 0:
            return render_template('myform.html', title='', form=form, prediction='?', confidence='?')
        X_all, prediction = get_prediction(tweet_objects)
        print('for user' + input_field + 'prediction = ' + prediction)
        # calculate confidence
        probas = clf.predict_proba(X_all)
        print('probas=', probas)
        confidence = round(probas.max(), 4)
        print('predicted %s with probability %.4f' % (prediction, confidence))
        top_features = print_top_features(X_all)[0:3]
        print(top_features)
        return render_template('myform.html', title='', form=form, tweets=tweets, prediction=prediction,
                               confidence=confidence, top_features=top_features)
    return render_template('myform.html', title='', form=form, prediction='?', confidence='?', top_features='?')

# @app.route('/', methods=['GET', 'POST'])
# @app.route('/index', methods=['GET', 'POST'])
# def test_api():
#     form = MyForm()
#     # result = None
#     data = request.get_json()
#     if data is not None:
#         print("get data: ", data)
#         screen_name = str(data.get('screen_name'))
#         print("screen_name: ", screen_name)
#         tweet_objects = [t for t in twapi._get_tweets('screen_name', screen_name, limit=200)]
#         tweets = [t['full_text'] for t in tweet_objects]
#         if len(tweet_objects) == 0:
#             return render_template('myform.html', title='', form=form, prediction='?', confidence='?')
#         X_all, prediction = get_prediction(tweet_objects)
#         print('for user' + screen_name + 'prediction = ' + prediction)
#         # calculate confidence
#         probas = clf.predict_proba(X_all)
#         print('probas=', probas)
#         confidence = round(probas.max(), 4)
#         print('predicted %s with probability %.4f' % (prediction, confidence))
#         top_features = print_top_features(X_all)[0:3]
#         print(top_features)
#         return render_template('myform.html', title='', form=form, tweets=tweets, prediction=prediction,
#                                confidence=confidence, top_features=top_features)
#     return render_template('myform.html', title='', form=form, prediction='?', confidence='?', top_features='?')

def get_prediction(tweet_objects):
    tweets = [t['full_text'] for t in tweet_objects]
    user = tweet_objects[0]['user']
    followers_count = user['followers_count']
    listed_count = user['listed_count']
    friends_count = user['friends_count']
    default_profile_image = int(user['default_profile_image'])
    default_profile = int(user['default_profile'])
    verified = int(user['verified'])
    statuses_count = user['statuses_count']
    name = user['name']
    screen_name = user['screen_name']

    feature_dicts = []
    features = get_user_features(tweets, tweets, len(tweet_objects), followers_count, listed_count, friends_count,
                                   default_profile_image, default_profile, statuses_count, verified, name, screen_name)
    feature_dicts.append(features)
    X_features = dict_vec.transform(feature_dicts)
    X_words = count_vec.transform([str(tweets)])
    # X_all = hstack([X_features, X_words]).tocsr()
    X_all = X_features
    scaled_X_all = scaler.transform(X_all)
    prediction = clf.predict(scaled_X_all)[0]
    print("in get prediction: ", prediction)
    return scaled_X_all, prediction

def print_top_features(X_all):
    coef = [-clf.coef_[0], clf.coef_[0]]
    features = dict_vec.get_feature_names() + count_vec.get_feature_names()
    # why was the first example labeled bot/human?
    top_features = []
    for i in np.argsort(coef[0][X_all[0].nonzero()[1]])[-1:-11:-1]:
        idx = X_all[0].nonzero()[1][i]
        print(features[idx])
        print(coef[0][idx])
        top_features.append((features[idx],coef[0][idx]))
    return top_features

if __name__ == "__main__":
    # from .app import app

    # from .app import routes
    app.run(port=5000, debug=True)
    sys.exit(main())  #from app import app pragma: no cover