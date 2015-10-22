
# coding: utf-8

# This notebook collects the XGBoost, SGDClassifiers and turns them into the same format as scikit-learn classifiers to make the use of them straightforward.

# In[95]:

import cPickle as pickle
import itertools
import json
import scipy as sp
from scipy.optimize import minimize
import numpy as np
from pandas import *
import pandas as pd
import warnings
from functools import partial
from operator import itemgetter
from sklearn.metrics import roc_curve, auc
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation, linear_model
#import xgboost as xgb
from sklearn import metrics
import time

#import mytimer
#mytimer=mytimer.Timer()

from sklearn.ensemble import RandomForestClassifier 
from sklearn.preprocessing import LabelEncoder
import operator
import matplotlib.pyplot as plt

get_ipython().magic(u'matplotlib inline')


# In[96]:

class XGBoostClassifier():
    def __init__(self, num_boost_round=40, **params):
        self.clf = None
        self.num_boost_round = num_boost_round
        self.params = params
        self.params.update({'objective': 'binary:logistic'})
 
    def fit(self, X, y, num_boost_round=None):
        num_boost_round = num_boost_round or self.num_boost_round
        dtrain = xgb.DMatrix(X.values, label=y)
        self.clf = xgb.train(params=self.params, dtrain=dtrain, num_boost_round=num_boost_round)
        return self
 
    def predict(self, X):
        Y = self.predict_proba(X.values)
        y = np.argmax(Y, axis=1)
        return y
 
    def predict_proba(self, X):
        ypreds = np.zeros((X.shape[0],2))
        dtest = xgb.DMatrix(X.values)
        ypreds[:,1] = self.clf.predict(dtest)
        ypreds[:,0] = 1- ypreds[:,1]        # return the proba for both classes
        return y
 
    def score(self, X, y):
        Y = self.predict_proba(X)
        return 1 / self.logloss(y, Y)
 
    def get_params(self, deep=True):
        return self.params
 
    def set_params(self, **params):
        if 'num_boost_round' in params:
            self.num_boost_round = params.pop('num_boost_round')
        if 'objective' in params:
            del params['objective']
        self.params.update(params)
        return self
    
    def logloss(self,y_true, Y_pred):
        label2num = dict((name, i) for i, name in enumerate(sorted(set(y_true))))
        return -1 * sum(math.log(y[label2num[label]]) if y[label2num[label]] > 0 else -np.inf                         for y, label in zip(Y_pred, y_true)) / len(Y_pred)


# In[97]:

class PAClassifier():
    def __init__(self, niter=100, loss='squared_hinge',C=1.0, SEED=123, **params):
        self.clf = None
        self.niter = niter
        self.SEED = SEED
        self.loss = loss
        self.C = C
        self.params = params
 
    def fit(self, X, y):
        self.clf=[]
        for i in range(self.niter):
            clf0= linear_model.PassiveAggressiveClassifier(loss=self.loss,C=self.C, random_state=i*self.SEED)
            clf0.fit(X, y)
            self.clf.append(clf0)
        return self    
 
    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
 
    def predict_proba(self, X):
        yprob = np.zeros((X.shape[0],2))
        for i in range(self.niter):
            yprob[:,1]=yprob[:,1] + self.clf[i].predict(X)
        yprob = yprob/ float(self.niter)
        yprob[:,0] = 1-yprob[:,1]
        return yprob
 
    def score(self, X, y):
        Y = self.predict_proba(X)
        return 1 / self.logloss(y, Y)
 
    def get_params(self, deep=True):
        return self.params
 
    def set_params(self, **params):
        self.params.update(params)
        return self
     
    def logloss(self,y_true, Y_pred):
        label2num = dict((name, i) for i, name in enumerate(sorted(set(y_true))))
        return -1 * sum(math.log(y[label2num[label]]) if y[label2num[label]] > 0 else -np.inf                         for y, label in zip(Y_pred, y_true)) / len(Y_pred)


# In[98]:

class SGDSVMClassifier():
    def __init__(self, niter=100, penalty='elasticnet',alpha=0.1,eta0=0.0, SEED=123, **params):
        self.clf = None
        self.niter = niter
        self.SEED = SEED
        self.penalty = penalty
        self.alpha = alpha
        self.eta0 = eta0
        self.params = params
 
    def fit(self, X, y):
        self.clf=[]
        for i in range(self.niter):
            clf0= linear_model.SGDClassifier(loss='hinge', penalty=self.penalty,                                              alpha=self.alpha,eta0=self.eta0, random_state=i*self.SEED)
            clf0.fit(X, y)
            self.clf.append(clf0)
        return self    
 
    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
 
    def predict_proba(self, X):
        yprob = np.zeros((X.shape[0],2))
        for i in range(self.niter):
            yprob[:,1]=yprob[:,1] + self.clf[i].predict(X)
        yprob = yprob/ float(self.niter)
        yprob[:,0] = 1-yprob[:,1]
        return yprob
 
    def score(self, X, y):
        Y = self.predict_proba(X)
        return 1 / self.logloss(y, Y)
 
    def get_params(self, deep=True):
        return self.params
 
    def set_params(self, **params):
        self.params.update(params)
        return self
     
    def logloss(self,y_true, Y_pred):
        label2num = dict((name, i) for i, name in enumerate(sorted(set(y_true))))
        return -1 * sum(math.log(y[label2num[label]]) if y[label2num[label]] > 0 else -np.inf                         for y, label in zip(Y_pred, y_true)) / len(Y_pred)







