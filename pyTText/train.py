from pyTText.utils import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.model_selection import *


class train(object):
    def __init__(self, df, session=None, target=None, testratio=0.3, stratifycol='sentiment', seed=100):

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
        self.seed = seed
        self.target = target
        self.stratify = stratifycol
        self.testratio = testratio
        self.sessions = {}

    def splitdata(self):
        for session, df in zip(self.sesslist, self.dflist):
            Xcol = df[[col for col in df.columns.tolist() if not self.target]]
            ycol = df[[self.target]]
            self.sessions[session] = Session()
            self.sessions[session].name = session
            self.sessions[session].X_train, self.sessions[session].y_train, self.sessions[session].X_test, self.sessions[session].y_test = train_test_split(Xcol, ycol, test_size=self.testratio, stratify=df[[self.stratify]], random_state=self.seed)

    def randomsearch(self):
        return None

    def train(self):
        return None