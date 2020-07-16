import argparse,os,sys
from pyTText.core import *
from pyTText.model import *
from pyTText.utils import *
from tabulate import tabulate
import random
import getpass

random.seed(100)
parser = argparse.ArgumentParser(description="pyTText: twitter sentiment analysis!")


def demo():
    print('{0}{1}{0}'.format('#'*20,'Load demo tweet dataset'))
    df = pd.read_csv('./data/tweets.csv')
    dft = df[['handle', 'text', 'is_retweet', 'original_author', 'time', 'lang', 'retweet_count', 'favorite_count']]
    print(tabulate(dft.head(), headers='keys', tablefmt='psql'))
    print('{0}{1}{0}'.format('#' * 20, 'Preprocessing dataset'))
    tlib = TextLibrary()
    pattern = tlib.patternlist()
    ttransform = TextTransform(pattern, backup='oritext')
    tweetdf = ttransform.preprocess(dft)
    print(tabulate(tweetdf.head(), headers='keys', tablefmt='psql'))
    print('{0}{1}{0}'.format('#' * 20, 'Scoring tweet using lexicon'))
    tweetscoredf = ttransform.opinionscore(tweetdf, remove='trump')
    print(tabulate(tweetscoredf.head(), headers='keys', tablefmt='psql'))
    print('{0}{1}{0}'.format('#' * 20, 'Creating DTM'))
    tokencol = ttransform.splituser(tweetscoredf)
    dtm = ttransform.dtm
    tr = TTrain(tokencol, target='sentiment', testratio=0.25)
    tr.splitdata(dtm=dtm)
    print(tabulate(tr.summarysentiment(), headers='keys', tablefmt='psql'))
    print('{0}{1}{0}'.format('#' * 20, 'Training Model'))
    tr.train()
    print(tabulate(tr.summarymetric(), headers='keys', tablefmt='psql'))
    print('{0}{1}{0}'.format('#' * 20, 'Inferencing Model'))
    ti = TInfer(tr.sessions)
    text = input("Enter any text/sentence: ")
    bestmodel = ti.sessions[random.choice(list(tr.sessions.keys()))].bestmodel
    print(ti.predict(text, bestmodel))


def demorefine():
    print('{0}{1}{0}'.format('#'*20,'Load demo tweet dataset'))
    df = pd.read_csv('./data/tweets.csv')
    dft = df[['handle', 'text', 'is_retweet', 'original_author', 'time', 'lang', 'retweet_count', 'favorite_count']]
    print(tabulate(dft.head(), headers='keys', tablefmt='psql'))
    print('{0}{1}{0}'.format('#' * 20, 'Preprocessing dataset'))
    tlib = TextLibrary()
    pattern = tlib.patternlist()
    ttransform = TextTransform(pattern, backup='oritext')
    tweetdf = ttransform.preprocess(dft)
    print(tabulate(tweetdf.head(), headers='keys', tablefmt='psql'))
    print('{0}{1}{0}'.format('#' * 20, 'Scoring tweet using lexicon'))
    tweetscoredf = ttransform.opinionscore(tweetdf, remove='trump')
    print(tabulate(tweetscoredf.head(), headers='keys', tablefmt='psql'))
    print('{0}{1}{0}'.format('#' * 20, 'Creating DTM'))
    tokencol = ttransform.splituser(tweetscoredf)
    dtm = ttransform.dtm
    tr = TTrain(tokencol, target='sentiment', testratio=0.25)
    tr.splitdata(dtm=dtm)
    print(tabulate(tr.summarysentiment(), headers='keys', tablefmt='psql'))
    tr.save_session()
    print('{0}{1}{0}'.format('#' * 20, 'Training Model'))
    tr.train()
    print(tabulate(tr.summarymetric(), headers='keys', tablefmt='psql'))
    print('{0}{1}{0}'.format('#' * 20, 'Inferencing Model with Twitter feed!'))
    ti = TInfer(tr.sessions)
    consumer_key = getpass.getpass('Enter twitter consumer_key api:')
    consumer_secret = getpass.getpass('Enter twitter consumer_secret api:')
    ttweet = TTweet(consumer_key, consumer_secret)

    for sess, obj in ti.sessions.items():
        for name, model in obj.models.items():
            ttweet.usertweet(sess, count=10)
            pred = ti.predict(ttweet.text, model.name, sess)
            print(tabulate(pred, headers='keys', tablefmt='psql'))


def tweetdemo():
    username = input('Enter twitter username:')
    ti = TInfer()
    print('{0}{1}{0}'.format('#' * 20, 'Loading pretrained data!'))
    ti.load_session()
    print('{0}{1}{0}'.format('#' * 20, 'Twitter scraping!'))
    consumer_key = getpass.getpass('Enter twitter consumer_key api:')
    consumer_secret = getpass.getpass('Enter twitter consumer_secret api:')
    ttweet = TTweet(consumer_key, consumer_secret)
    ttweet.usertweet(username, count=10)

    for sess, obj in ti.sessions.items():
        for name, model in obj.models.items():
            pred = ti.predict(ttweet.text, model.name, sess)
            print(tabulate(pred, headers='keys', tablefmt='psql'))


parser.add_argument('--demoextended',action="store_true", help="Trump/Clinton dataset with GridSeach tuning as well Twitter live demo scraping.")
parser.add_argument('--twitterdemo',action="store_true", help="Twitter live demo scraping.")
parser.add_argument('--demo', action="store_true", help="Simple demo training, testing and inferencing.")

args = parser.parse_args()

if args.twitterdemo:
    tweetdemo()
elif args.demo:
    demo()
elif args.demoextended:
    demorefine()
else:
    print('Invalid Input. Run --help for more info')
