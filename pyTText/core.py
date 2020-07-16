import tweepy
import pandas as pd
from .utils import *
from .model import *


class TTweet(object):

    def __init__(self,consumerkey,consumersecret):
        self.ckey = consumerkey
        self.csecret = consumersecret
        auth = tweepy.AppAuthHandler(self.ckey, self.csecret)
        self.api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
        self.rawtweet, self.tweet = None, None
        self.text = []
        self.session = 'Unknown'

    def usertweet(self, username, retweet=False, count=200):

        alltweets = []
        new_tweets = self.api.user_timeline(screen_name=username, count=count)
        alltweets.extend(new_tweets)
        dft = None
        if not retweet:
            dft = pd.DataFrame([{'handle': tweet.user.screen_name, 'time': tweet.created_at, 'lang': tweet.lang,
                                 'original_author': tweet.author.screen_name, 'favorite_count': tweet.favorite_count,
                                 'retweet_count': tweet.retweet_count, 'is_retweet': tweet.retweeted,
                                 'text': tweet.text.encode("utf-8").decode('utf-8')} for tweet in alltweets if not tweet.retweeted])
        else:
            dft = pd.DataFrame([{'handle': tweet.user.screen_name, 'time': tweet.created_at, 'lang': tweet.lang,
                                 'original_author': tweet.author.screen_name, 'favorite_count': tweet.favorite_count,
                                 'retweet_count': tweet.retweet_count, 'is_retweet': tweet.retweeted,
                                 'text': tweet.text.encode("utf-8").decode('utf-8')} for tweet in alltweets])
        count = len(dft) if len(dft) > count else count
        dft = dft.head(count)
        self.rawtweet = dft
        tlib = TextLibrary()
        pattern = tlib.patternlist()
        ttransform = TextTransform(pattern, backup='oritext')
        self.tweet = ttransform.preprocess(dft)
        self.text = self.tweet['text'].tolist()
        self.session = username

        return dft

