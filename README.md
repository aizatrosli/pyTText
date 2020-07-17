# pyTText #

#### pyThoughtText ####

Twitter sentiment analysis using opinion lexicon and DTM. 
Compatible with any sklearn models as well high level Keras API *(keras.wrappers.scikit_learn)* and tensorflow 2.0 *(tf.estimator)*

### Requirements ###

* pip install nltk, tqdm, tabulate, tweepy

### How to run ###
```bash
python3 pyttext.py

usage: pyttext.py [-h] [--demoextended] [--twitterdemo] [--demo]

pyTText: twitter sentiment analysis!

optional arguments:
  -h, --help      show this help message and exit
  --demoextended  Trump/Clinton dataset with GridSeach tuning as well Twitter
                  live demo scraping.
  --twitterdemo   Twitter live demo scraping.
  --demo          Simple demo training, testing and inferencing.
```

[![asciicast](https://asciinema.org/a/347916.svg)](https://asciinema.org/a/347916)

for more detail explaination, please refer [pyTText.md](pyTText.md) or on our blog post [Sentiment analysis on Trump and Hillary tweets](https://abualfateh2901.wixsite.com/afsaanalytics/post/sentiment-analysis-on-trump-and-hillary-tweets).

### Demo with Google Colab ###

[pyttext_demo.ipynb](https://colab.research.google.com/drive/1dEQiLfAi4YE2Z9kC1lzHdjQWnUSqIc8P?usp=sharing)

### Contact Us ###
[ASFA Analytics](https://abualfateh2901.wixsite.com/afsaanalytics)