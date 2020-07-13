from datetime import datetime
import os, sys, time
import nltk, re, string

class TextLibrary(object):
    def __init__(self):
        self.hashtag = r'#.*?(?=\s|$)'
        self.mention = r'@.*?(?=\s|$)'
        self.digit = r'\d+'
        self.hyperlink = r'^https?:\/\/.*[\r\n]*'
        self.specialchar = r'\W'
        self.punctuation = r'[^\w\s]'
        self.underscore = r'_'

    def patternlist(self):
        return self.__dict__

    def reset(self):
        return self.__init__()

    def opinion_lexicon(self, opinion=None):
        from nltk.corpus import opinion_lexicon
        nltk.download('opinion_lexicon', quiet=True)
        if opinion == 'positive':
            return opinion_lexicon.positive()
        elif opinion == 'negative':
            return opinion_lexicon.negative()
        else:
            return opinion_lexicon.words()


class TextTransform(object):
    def __init__(self, pattern, backup=None, targetcol='text', lang='english', cases='lower', hashtag=True, mention=True):
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
        self.stopword = stopwords.words(lang)

    def process(self, df, textcol):
        if self.backup is not None:
            df[self.backup] = df[textcol]
        if self.hashtag:
            df['hashtag'] = df[textcol].str.findall(self.pattern['hashtag'])
            df[textcol] = df[textcol].str.replace(self.pattern['hashtag'], " ")
        if self.mention:
            df["accounts"] = df[textcol].str.findall(self.pattern['mention'])
            df[textcol] = df[textcol].str.replace(self.pattern['mention'], " ")
        df[textcol] = df[textcol].str.lower() if self.cases == 'lower' else df[textcol]
        for tag, pat in self.pattern.items():
            df[textcol] = df[textcol].str.replace(pat, " ")
        df[textcol] = df[textcol].apply(lambda x: " ".join([i for i in x.split() if i not in self.stopword]))
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
        df = df.apply(self.sentencelookup, scorename = 'positive', scoreset = self.positive, axis=1)
        df = df.apply(self.sentencelookup, scorename = 'negative', scoreset = self.negative, axis=1)
        df['score'] = df["positive_count"] - df["negative_count"]
        df = df.apply(self.sentimentscore, axis=1)
        return df
