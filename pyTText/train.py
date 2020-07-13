from pyttext.utils import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.model_selection import *
from sklearn.naive_bayes import *
from sklearn.neural_network import *
from sklearn.ensemble import *
from sklearn.tree import *
from sklearn.svm import *


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
        self.seed = seed
        self.target = target
        self.stratify = stratifycol
        self.testratio = testratio
        self.sessions = {}
        self.model = None

    def splitdata(self):
        '''
        split data. if there is multiple session then split by session.
        :return:
        '''
        for session, df in zip(self.sesslist, self.dflist):
            Xcol = df[[col for col in df.columns.tolist() if col not in [self.target, 'text']]]
            ycol = df[[self.target]]
            self.sessions[session] = Session()
            self.sessions[session].name = session
            self.sessions[session].X_train, self.sessions[session].X_test, self.sessions[session].y_train, self.sessions[session].y_test = train_test_split(Xcol, ycol, test_size=self.testratio, stratify=df[[self.stratify]], random_state=self.seed)

    def defaultmodel(self, model):
        '''
        sample model format :
        self.model = {
                'GaussianNaiveBayes': {'model': GaussianNB(), 'params': {'var_smoothing': [1e-9, 1e-7]}},
                'SVM': {'model': SVC(), 'params': {'kernel': ['rbf', 'poly'], 'C': [1.0, 1.5, 4]}},
                }
        if there is no model specify, use default models.
        :param model: dictionary model contains compatible sklearn's model with its hyperparam
        :return:
        '''
        if type(model) is not dict and model is not None:
            raise TypeError('Dict of sklearn compatible models')
        if self.model is None and model is None:
            self.model = {
                'GaussianNaiveBayes': {'model': GaussianNB(), 'params': {'var_smoothing': [1e-9, 1e-7]}},
                'SVM': {'model': SVC(), 'params': {'kernel': ['rbf', 'poly'], 'C': [1.0, 1.5, 4]}},
                'MultiLayerPerceptron': {'model': MLPClassifier(),
                                         'params': {'activation': ['tanh', 'relu'], 'solver': ['sgd', 'adam'],
                                                    'alpha': [0.00001, 0.000001, 0.01]}},
                'DecisionTree': {'model': DecisionTreeClassifier(),
                                 'params': {'criterion': ['gini', 'entropy'], 'max_depth': [None, 100, 32, 5],
                                            'min_samples_leaf': [5, 1, 10]}}
            }
        else:
            self.model = model

    def randomsearchtrain(self, model=None, n_iter=10, n_jobs=None):
        '''
        randomize for hyperparameter tuning
        :param model: please refer tu defaultmodel()
        :param n_iter: iteration for each model
        :param n_jobs: cpu used for parallel compute. default None == 1 cpu, for all cpu usage please use -1
        :return:
        '''
        self.defaultmodel(model)
        if not self.sessions:
            self.splitdata()
            #raise Exception('Please execute splitdata!')
        for session, data in self.sessions.items():
            for name, model in self.model.items():
                print('Session:{0}\tModel:{1}'.format(session, name))
                self.sessions[session].model[name] = RandomizedSearchCV(model['model'], param_distributions=model['params'], n_iter=n_iter, n_jobs=n_jobs)
                self.sessions[session].model[name].fit(self.sessions[session].X_train, self.sessions[session].y_train)
                self.sessions[session].model[name].params = self.sessions[session].model[name].best_params_

    def train(self, model=None, randomparams=False):
        '''
        default train
        :param model: please refer tu defaultmodel()
        :param randomparams: pick one from list of hyperparams using ParameterSampler. default False.
        :return:
        '''
        self.defaultmodel(model)
        if not self.sessions:
            self.splitdata()
            #raise Exception('Please execute splitdata!')
        for session, data in self.sessions.items():
            for name, model in self.model.items():
                print('Session:{0}\tModel:{1}'.format(session, name))
                self.sessions[session].model[name] = model['model']
                if randomparams and model['params']:
                    self.sessions[session].model[name].params = ParameterSampler(model['params'], n_iter=1, random_state=self.seed)
                    self.sessions[session].model[name].set_params(self.sessions[session].params)
                self.sessions[session].model[name] = model['model']
                self.sessions[session].model[name].fit(self.sessions[session].X_train, self.sessions[session].y_train)

