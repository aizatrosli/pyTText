import argparse,os,sys
from pyTText.core import *
from pyTText.model import *
from pyTText.utils import *
import getpass

parser = argparse.ArgumentParser(description="pyTText: twitter sentiment analysis!")


def demo():
    print('{0}{1}{0}'.format('#'*20,'Load demo tweet dataset'))
    df = pd.read_csv('./data/tweets.csv')
    dft = df[['handle', 'text', 'is_retweet', 'original_author', 'time', 'lang', 'retweet_count', 'favorite_count']]
    print(dft.head())
    print('{0}{1}{0}'.format('#' * 20, 'Preprocessing dataset'))
    tlib = TextLibrary()
    pattern = tlib.patternlist()
    ttransform = TextTransform(pattern, backup='oritext')
    tweetdf = ttransform.preprocess(dft)
    print(tweetdf.head().to_markdown())
    print('{0}{1}{0}'.format('#' * 20, 'Scoring tweet using lexicon'))
    tweetscoredf = ttransform.opinionscore(tweetdf, remove='trump')
    print(tweetscoredf.head().to_markdown())
    print('{0}{1}{0}'.format('#' * 20, 'Creating DTM'))
    tokencol = ttransform.splituser(tweetscoredf)
    dtm = ttransform.dtm
    tr = TTrain(tokencol, target='sentiment', testratio=0.25)
    tr.splitdata(dtm=dtm)
    tr.summarysentiment()
    print('{0}{1}{0}'.format('#' * 20, 'Training Model'))
    tr.train()
    print(pd.DataFrame(tr.model).to_markdown())
    print('{0}{1}{0}'.format('#' * 20, 'Inferencing Model'))
    ti = TInfer(tr.sessions)
    text = input("Enter any text/sentence: ")
    bestmodel = ti.sessions['realDonaldTrump'].bestmodel
    print(ti.predict(text, bestmodel))


def tweetdemo():
    username = input('Enter twitter username:')
    ti = TInfer()
    ti.load_session()
    consumer_key = getpass.getpass('Enter twitter consumer_key api:')
    consumer_secret = getpass.getpass('Enter twitter consumer_secret api:')
    ttweet = TTweet(consumer_key, consumer_secret)
    ttweet.usertweet(username, count=10)

    for sess, obj in ti.sessions.items():
        for name, model in obj.models.items():
            pred = ti.predict(ttweet.text, model.name, sess)
            print(pred.to_markdown())


group1 = parser.add_mutually_exclusive_group(required=True)
group1.add_argument('--twitterdemo',action="store_true", help="Twitter live demo scraping.")
group1.add_argument('--demo',action="store_false", help="Simple demo training, testing and inferencing.")
args = parser.parse_args()

if args.twitterdemo:
    tweetdemo()
elif not args.demo:
    demo()
