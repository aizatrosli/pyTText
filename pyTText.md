# pyTText #

### core.py ###
#### TTweet ####
* **tweepy** api handler to pull handle tweet from timeline.
* generate preprocessed text from **utils.py** for other usage (training or inferencing).
* **consumer key** and **consumer secret** api are needed for this feature. limit up to **200 tweets** for single api calls.
```python
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
```

### model.py ###
#### TTrain ####
* *session* is based on **how many datasets** that have been splitted during preprocessing data.
* every *session* has different model trained. this is suitable for training sentiment model pattern of the tweet handle itself.
```python
class TTrain(object):
    def __init__(self, df, session=None, target=None, testratio=0.3, stratifycol='sentiment', seed=100):
        '''

        :param df: dataframe with text for sentiment analysis
        :param session: name of session if multiple session, use split with ','
        :param target: test
        :param testratio: ratio size of test sample
        :param stratifycol: default using sentiment column
        :param seed: for randomize stuff
        '''
        if type(df) is dict:
            self.dflist = [val for key, val in df.items()]
            if session is not None:
                if len(self.dflist) == len(session.split(',')):
                    self.sesslist = session.spit(',')
            else:
                self.sesslist = list(df.keys())
        elif type(df) is list:
            self.dflist = df
            self.sesslist = ['task_tweet%d' % i for i, j in enumerate(self.dflist)]
        elif type(df) is pd.core.frame.DataFrame:
            self.dflist = [df]
            self.sesslist = ['task_tweet'] if session is None else session
        else:
            raise TypeError('Oi wrong type MF')
```
* using sklearn for training libraries. this can be expanded with other libraries such as Keras API and Tensorflow 2.0
```python
    self.model = {
        'GaussianNaiveBayes': {'model': GaussianNB(), 'params': {'var_smoothing': [1e-9, 1e-7]}},
        'Support Vector Machines': {'model': SVC(), 'params': {'kernel': ['rbf', 'poly'], 'C': [1.0, 1.5, 4]}},
        'MultiLayerPerceptron': {'model': MLPClassifier(),
                                 'params': {'activation': ['tanh', 'relu'], 'solver': ['sgd', 'adam'],
                                            'alpha': [0.00001, 0.000001, 0.01]}},
        'DecisionTree': {'model': DecisionTreeClassifier(),
                         'params': {'criterion': ['gini', 'entropy'], 'max_depth': [None, 100, 32, 5],
                                    'min_samples_leaf': [5, 1, 10]}}
    }
```
* split data for big dataset with different handle. it split into multiple 'sessions'.
* split train and test data based on test ratio size (*test=0.3*). by default the stratify used for sentiment column.
```python
    splitdata(self, test=None, dtm=None)
```
* basic train function. *params* can be randomized if it is define in *self.model*.
```python
train(self, model=None, randomparams=False)
```
* *gridsearchtrain* is **hyperparameter tunning** function based on *params* in *self.model*.
```python
gridsearchtrain(self, model=None, n_jobs=None)
```
* this metric includes **accuracy_score**,**f1_score**,**recall_score** and **precision_score**. metric function executed on both *train* and *gridseachtrain*. metric results can be found in *session*.
```python
metric(self, y_actual, y_pred)
```
* for comparison metrics between models and session.

```python
summarymetric(self)
```
* summaries number of sentiment for each session and data split.
```python
summarysentiment(self)
```
* load and save session for inferencing or retrain the models.
```python
save_session(self,picklename='ttsessions.pkl')
load_session(self,picklename='ttsessions.pkl')
```

#### TInfer ####

* run prediction with new dataset. if session is undefined it will randomize based on pretrained session.

```python
class TInfer(object):
    def __init__(self, session=None, seed=100):
        random.seed(seed)
        self.sessions = session

    def load_session(self,picklename='ttsessions.pkl'):
        import pickle as pkl
        with open(picklename, 'rb') as fp:
            self.sessions = None
            self.sessions = pkl.load(fp)
            return self.sessions

    def predict(self, text, modelname, session=None):
        '''
        predict text sentiment. modelname is needed
        :param text: string of text or list of text
        :param modelname: model or use session.model
        :param session: session name. default random pick session
        :return: sentiment result
        '''
        if type(text) is str:
            text = [text]
        if session is None:
            session = random.choice(list(self.sessions.keys()))
        vectorizer = self.sessions[session].dtm
        vec = vectorizer.transform(text)
        dtm = pd.DataFrame(vec.toarray(), columns=vectorizer.get_feature_names())
        pred = self.sessions[session].models[modelname].model.predict(dtm)
        preddf = pd.DataFrame()
        preddf['text'] = text
        preddf['sentiment'] = ["negative" if i == 3 else 'positive' if i == 2 else "neutral" for i in pred]
        preddf['session'] = str(session)
        preddf['model'] = str(modelname)
        return preddf
```

### utils.py ###
#### TextLibrary ####

* these are all regex patterns for preprocessing text.

```python
    self.hashtag = r'#.*?(?=\s|$)'
    self.mention = r'@.*?(?=\s|$)'
    self.digit = r'\d+'
    #self.hyperlink = r'^https?:\/\/.*[\r\n]*'
    self.specialchar = r'\W'
    self.punctuation = r'[^\w\s]'
    self.underscore = r'_'
```
* using nltk library to load opinion lexicon dictionary. 

```python
    opinion_lexicon(self, opinion=None)
```

#### TextTransform ####

* text preprocessing function; extract and strip the original text with *TextLibrary* pattern. 
```python
    def process(self, df, textcol, retweet):
        '''
        process dataframe
        :param df: target dataframe
        :param textcol: target column which 'text'
        :return:
        '''
        if self.backup is not None:
            df[self.backup] = df[textcol]
        if self.hashtag:
            df['hashtag'] = df[textcol].str.findall(self.pattern['hashtag'])
            df[textcol] = df[textcol].str.replace(self.pattern['hashtag'], " ")
        if self.mention:
            df["accounts"] = df[textcol].str.findall(self.pattern['mention'])
            df[textcol] = df[textcol].str.replace(self.pattern['mention'], " ")
        df[textcol] = df[textcol].str.lower() if self.cases == 'lower' else df[textcol]
        df[textcol] = df[textcol].apply(lambda x:re.split(r'https:\/\/.*',str(x))[0])
        for tag, pat in self.pattern.items():
            df[textcol] = df[textcol].str.replace(pat, " ")
        df[textcol] = df[textcol].progress_apply(lambda x: " ".join([i for i in x.split() if i not in self.stopword]))
        if not retweet:
            df = df[df["is_retweet"] == False].reset_index()
        return df
```
* comparing each row of text with lexicon dictionaries list.
```python
    sentencelookup(self, df, scorename, scoreset)
```
* score calculated based on difference of count of sentiments.
```python
    def sentimentscore(self,df):
        '''
         Score based on total of positive words - total of negative words
        :param df: target dataframe
        :return: process dataframe
        '''
        df['sentiment'] = "negative" if df['score'] < 0 else "positive" if df['score'] > 0 else "neutral"
        return df
```
* data can be split by handle for better accuracy on predicting the sentiment for text.
```python
    splituser(self, df, tokenize=True, usercol='handle')
```
* datasets of text that have been preprocessed need to be changed into vector or *document term matrix (DTM)*. 
```python
    def tokenizer(self, df):
        '''
        split text into document matrix term
        :param df: target dataframe
        :return: dataframe document matrix term
        '''
        from sklearn.feature_extraction.text import CountVectorizer
        documents = df[self.target].tolist()
        if 'sentiment' not in df.columns.tolist():
            raise Exception('MF please run opinion score first!')
        vectorizer = CountVectorizer()
        vec = vectorizer.fit_transform(documents)
        self.dtm = vectorizer
        dtm = pd.DataFrame(vec.toarray(), columns=vectorizer.get_feature_names())
        dtm = df[[self.target,"sentiment"]].merge(dtm, left_index=True, right_index=True, how="left")
        dtm["sentiment"] = dtm["sentiment"].map({"neutral" : 1,"positive" : 2,"negative" : 3})
        dtm = dtm.rename(columns={"text_x": "text"})
        return dtm
```

#### Session & Model ####

* to handle session data and others
```python
class Session(object):
    def __init__(self):
        self.name = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.models = {}
        self.params = None
        self.dtm = None
```

* to handle model data and results.
```python
class Model(object):
    def __init__(self):
        self.model=None
        self.name=None
        self.params=None
        self.metric={}
        self.actual=[]
        self.predict=[]
```