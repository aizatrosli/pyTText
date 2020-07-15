import argparse,os,sys
from pyTText.core import *
from pyTText.model import *
from pyTText.utils import *

df = pd.read_csv(r'https://raw.githubusercontent.com/wimlds/election-data-hackathon/master/clinton-trump-tweets/data/tweets.csv')
dft = df[[ 'handle', 'text', 'is_retweet', 'original_author', 'time', 'lang', 'retweet_count', 'favorite_count']]

pat = TextLibrary()
ttransform = TextTransform(pat.patternlist(), backup='oritext')
ndf = ttransform.preprocess(dft)
nndf = ttransform.opinionscore(ndf, remove='trump')
nnndf = ttransform.tokenizer(nndf)
tr = TTrain(nnndf, target='sentiment')
tr.splitdata()
tr.train()
print(tr.sessions['task_tweet'].model)


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
    print(tweetdf.head())
    print('{0}{1}{0}'.format('#' * 20, 'Scoring tweet using lexicon'))
    tweetscoredf = ttransform.opinionscore(tweetdf, remove='trump')
    print(tweetscoredf.head())
    print('{0}{1}{0}'.format('#' * 20, 'Creating DTM'))
    tokencol = ttransform.splituser(tweetscoredf)
    dtm = ttransform.dtm
    tr.summarysentiment()
    print('{0}{1}{0}'.format('#' * 20, 'Training Model'))
    tr.train()
    print(pd.DataFrame(tr.model))
    print('{0}{1}{0}'.format('#' * 20, 'Inferencing Model'))
    ti = TInfer(tr.sessions)
    text = input("Enter any text/sentence: ")
    bestmodel = ti.sessions['realDonaldTrump'].bestmodel
    print(ti.predict(text, bestmodel))


def tweetdemo(username):
    ti = TInfer()
    ti.load_session()
    consumer_key = input('Enter twitter consumer_key api:')
    consumer_secret = input('Enter twitter consumer_secret api:')
    ttweet = TTweet(consumer_key, consumer_secret)
    ttweet.usertweet(username, count=10)

    for sess, obj in ti.sessions.items():
        for name, model in obj.models.items():
            pred = ti.predict(ttweet.text, model.name, sess)
            print(pred)