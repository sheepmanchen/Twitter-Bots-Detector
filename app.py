import click
from detector import credentials_path, clf_path
from detector.model import train
from detector.mytwitter import Twitter
import sys
# from detector.app import get_prediction
from detector.app import get_prediction
import pickle

twapi = Twitter(credentials_path)
clf, count_vec, dict_vec, scaler = pickle.load(open(clf_path, 'rb'))

@click.group()
def main(args=None):
    """Console script for osna."""
    return 0

@main.command('web')
@click.option('-t', '--twitter-credentials', required=False, type=click.Path(exists=True), show_default=True,
              default=credentials_path, help='a json file of twitter tokens')
@click.option('-p', '--port', required=False, default=5000, show_default=True, help='port of web server')
def web(twitter_credentials, port):
    # from detector.app import app
    from detector.app import app
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

if __name__ == "__main__":
    from detector.app import app
    app.run(port=5000, debug=True)
    sys.exit(main())  #from app import app pragma: no cover