from datetime import datetime
import pandas as pd
import os, sys, time
import nltk, re, string


class TextLibrary(object):
    """
    All patterns and lexicon dictionaries are here duh.
    """
    def __init__(self):
        self.hashtag = r'#.*?(?=\s|$)'
        self.mention = r'@.*?(?=\s|$)'
        self.digit = r'\d+'
        #self.hyperlink = r'^https?:\/\/.*[\r\n]*'
        self.specialchar = r'\W'
        self.punctuation = r'[^\w\s]'
        self.underscore = r'_'

    def patternlist(self):
        return self.__dict__

    def reset(self):
        return self.__init__()

    def opinion_lexicon(self, opinion=None):
        '''
        download lexicon dictionaries from nltk library
        :param opinion: positive or negative
        :return:
        '''
        from nltk.corpus import opinion_lexicon
        nltk.download('opinion_lexicon', quiet=True)
        if opinion == 'positive':
            return opinion_lexicon.positive()
        elif opinion == 'negative':
            return opinion_lexicon.negative()
        else:
            return opinion_lexicon.words()


class TextTransform(object):
    """
    This is for text processing, including sentiment scoring using TextLibrary references
    """
    def __init__(self, pattern, backup=None, targetcol='text', lang='english', cases='lower', hashtag=True, mention=True):
        import pandas as pd
        from tqdm.auto import tqdm
        tqdm.pandas()
        import nltk
        nltk.download('stopwords', quiet=True)
        from nltk.corpus import stopwords
        self.positive, self.negative = None, None
        self.target = targetcol
        self.cases = cases
        self.backup = backup
        self.pattern = pattern
        self.hashtag = hashtag
        self.mention = mention
        self.dtm = None
        self.stopword = stopwords.words(lang)

    def process(self, df, textcol):
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
        return df

    def preprocess(self, df):
        return self.process(df.copy(), textcol=self.target)

    def sentencelookup(self, df, scorename, scoreset):
        '''
        lexicon dictionary compare with text input
        :param sentstr: dataframe series for string
        :param scorename: for new series references
        :param scoreset: list of lexicon strings
        :return: dataframe with negative & positive string
        '''
        df[scorename] = list(set(df[self.target].split()).intersection(scoreset))
        df[scorename+'_count'] = len(df[scorename])
        return df

    def sentimentscore(self,df):
        '''
         Score based on total of positive words - total of negative words
        :param df: target dataframe
        :return: process dataframe
        '''
        df['sentiment'] = "negative" if df['score'] < 0 else "positive" if df['score'] > 0 else "neutral"
        return df

    def opinionscore(self, df, remove=None):
        '''

        :param df: target dataframe
        :param remove: unwanted string that match with dictionary. eg. people's name
        :return: dataframe with score
        '''
        tl = TextLibrary()
        remove = remove.split(',')
        self.positive = set(tl.opinion_lexicon('positive')) if remove is None else set(tl.opinion_lexicon('positive')).difference(set(remove))
        self.negative = set(tl.opinion_lexicon('negative')) if remove is None else set(tl.opinion_lexicon('negative')).difference(set(remove))
        df = df.progress_apply(self.sentencelookup, scorename = 'positive', scoreset = self.positive, axis=1)
        df = df.progress_apply(self.sentencelookup, scorename = 'negative', scoreset = self.negative, axis=1)
        df['score'] = df["positive_count"] - df["negative_count"]
        df = df.progress_apply(self.sentimentscore, axis=1)
        return df

    def splituser(self, df, tokenize=True, usercol='handle'):
        '''
        split df based on tweeter
        :param df: target dataframe
        :param usercol: column name for tweeter/handler
        :return: list of df with tweeter
        '''
        dfs = dict(tuple(df.groupby(usercol)))
        if not tokenize:
            return {x:dfs[x].reset_index() for x in dfs}
        else:
            return {x:self.tokenizer(dfs[x].reset_index()) for x in dfs}

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



class Model(object):
    def __init__(self):
        self.model=None
        self.name=None
        self.params=None
        self.metric={}
        self.actual=[]
        self.predict=[]

    def __eq__(self, other):
        if not isinstance(other, Model):
            return False
        if self.model == other.model and self.name == other.name and self.predict == other.predict:
            return True
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)


class Session(object):
    def __init__(self):
        '''
        class method for training session
        '''
        self.name = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.models = {}
        self.params = None
        self.dtm = None

    def summary(self):
        '''
        summary for per session
        :return:
        '''
        print('#' * 100)
        print('#{}#'.format(' ' * 98))
        print('#{}{}#'.format(str(self.name).upper(), ' ' * (98 - len(str(self.name)))))
        print('#{}#'.format(' ' * 98))
        print('#' * 100)
        for key, val in self.__dict__.items():
            print('#' * 50)
            if type(val) is list:
                print('\n## {}\t:{}\n'.format(str(key), ','.join(val)))
            elif type(val) is pd.core.frame.DataFrame:
                print('\n## {}\t:\n{}\n'.format(str(key), self.sumdf(val)))
            else:
                print('\n## {}\t:{}\n'.format(str(key), val))

    def sumdf(self, df):
        summarydf = pd.DataFrame(columns=df.columns)
        summarydf.loc["dtype"] = [df[col].dtype for col in df.columns.tolist()]
        summarydf.loc["nunique"] = [df[col].nunique() for col in df.columns.tolist()]
        summarydf.loc["unique"] = [df[col].unique() for col in df.columns.tolist()]
        summarydf.sort_values(by=['nunique'], axis=1)
        return summarydf.T

    def save(self):
        return None

    def load(self):
        return None