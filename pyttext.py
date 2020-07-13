import pandas as pd
from pyTText.utils import *
from pyTText.train import *

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