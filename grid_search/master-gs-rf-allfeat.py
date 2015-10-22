
# coding: utf-8

# In[79]:

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from scipy import stats,sparse
from sklearn.base import TransformerMixin
from datetime import datetime as dt
from math import isnan
from numpy import ma
import cPickle as pickle
import xgboost as xgb
import time
from pandas import *
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier 


# In[80]:

import json
from sklearn.metrics import roc_curve, auc
from re import sub
from collections import defaultdict


# In[81]:

from sklearn.cross_validation import StratifiedKFold,cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report,confusion_matrix


# In[82]:

class XGBoostClassifier():
    def __init__(self, num_boost_round=40, **params):
        self.clf = None
        self.num_boost_round = num_boost_round
        self.params = params
        self.params.update({'objective': 'binary:logistic'})
 
    def fit(self, X, y, num_boost_round=None):
        num_boost_round = num_boost_round or self.num_boost_round
        dtrain = xgb.DMatrix(X, label=y)
        self.clf = xgb.train(params=self.params, dtrain=dtrain, num_boost_round=num_boost_round)
        return self
 
    def predict(self, X):
        Y = self.predict_proba(X)
        y = np.argmax(Y, axis=1)
        return y
 
    def predict_proba(self, X):
        ypreds = np.zeros((X.shape[0],2))
        dtest = xgb.DMatrix(X)
        ypreds[:,1] = self.clf.predict(dtest)
        ypreds[:,0] = 1- ypreds[:,1]        # return the proba for both classes
        return ypreds
 
    def score(self, X, y):
        Y = self.predict_proba(X)
        return 1 / self.logloss(y, Y)
 
    def get_params(self, deep=True):
        return self.params
 
    def set_params(self, **params):
#        if 'num_boost_round' in params:
#            self.num_boost_round = params.pop('num_boost_round')
        if 'objective' in params:
            del params['objective']
        self.params.update(params)
        return self
    
    def logloss(self,y_true, Y_pred):
        label2num = dict((name, i) for i, name in enumerate(sorted(set(y_true))))
        return -1 * sum(math.log(y[label2num[label]]) if y[label2num[label]] > 0 else -np.inf for y, label in zip(Y_pred, y_true)) / len(Y_pred)


# ## 0. Load data

# In[83]:

LocalTest=False           # whether to do a local test
SelectedFeature=False    # whether to use selected features
nrows=1000


# In[84]:

if LocalTest:
    trainfile = 'C:/Huaixiu/Kaggle/GridSearch/data/train-5000.csv'
    xtrain = read_csv(trainfile,nrows=nrows)
    ytrain = xtrain['target']
    
    xtrain = xtrain.ix[:,1:-1]

else:    
    X=np.load('../data/nxtrain_standard_original0.npy')
    X1= np.load('../data/nxtrain_standard_derived0.npy')
    X2=pickle.load(open("../data/time_series_derived_train2.dat","rb"))
    X3=pickle.load(open("../data/time_series_original_train2.dat","rb"))
    X4=pickle.load(open("../data/cat_numeric_th60_train2.dat","rb"))
    ytrain=pickle.load(open("../data/ytrain2.dat","rb"))
    xtrain=np.hstack((X,X1,X2,X3,X4))
    del X,X1,X2,X3,X4
    
    if SelectedFeature:
        with open('../data/XGB_335Features Oct172015_044255_AUC_0p76115.p', 'rb') as fid:
            xgb_goodfeat = pickle.load(fid)
    
        good_features=list(xgb_goodfeat)
        xtrain = xtrain[:,good_features]


# In[85]:

print(xtrain.shape, ytrain.shape)


# ##2. GridSearch using RandomForest

# In[91]:

if LocalTest:
    param_rf = {'n_estimators': [200],'max_depth':[20],'n_jobs': [1],'max_features':['auto'],'min_samples_leaf':[1,3]}        
else:    
    param_rf = {'n_estimators': [111111],
            'max_depth':[222222],
         'n_jobs': [-1],
         'max_features':[333333],
        'min_samples_leaf':[444444]}


# In[92]:

start_time=time.clock()

print('Starting GridSearch using RandomForest...')
clf_rf = RandomForestClassifier(random_state =100)
gs_rf = GridSearchCV(clf_rf,param_grid = param_rf,cv = StratifiedKFold(ytrain,n_folds = 3),scoring='roc_auc', n_jobs = -1,verbose = 2)
gs_rf.fit(xtrain,ytrain)

total_time=time.clock()-start_time
print('Completed GridSearch using RandomForest')
print('Total running time is %d seconds\n' %total_time)


# In[93]:

print 'Best AUC Score of RF is {}'.format(gs_rf.best_score_)
print 'Best parameters set of RF:'
best_param_rf = gs_rf.best_estimator_.get_params()
for param_name in sorted(best_param_rf.keys()):
    print '\t%s: %r' % (param_name,best_param_rf[param_name])


# ###dump the model into pickle

# In[94]:

rf_opt = gs_rf.best_estimator_

outputfile='psAAAABBBBCCCCDDDD'

with open('rf_opt_allfeat_'+outputfile+ '_AUC_' + '0p'+ str(int(gs_rf.best_score_*1e5)) + '.pkl', 'wb') as fid:
    pickle.dump(rf_opt, fid,protocol = 2)
    
with open('rf_bestparam_allfeat_'+outputfile+ '_AUC_' + '0p'+ str(int(gs_rf.best_score_*1e5))  + '.pkl', 'wb') as fid:
    pickle.dump(best_param_rf, fid,protocol = 2) 



# ##3. make prediction on the test set

# In[96]:



clf1 = RandomForestClassifier(random_state =100)
clf1.set_params(**best_param_rf)
RF = clf1.fit(xtrain,ytrain)

del xtrain
del ytrain

with open('../data/xtest_ID.pkl','rb') as fid:
    test_ID = pickle.load(fid)

    # load test data set
if LocalTest:
    testfile = 'C:/Huaixiu/Kaggle/GridSearch/data/train-5000.csv'
    xtest = read_csv(testfile,nrows=nrows)
    xtest = xtest.ix[:,1:-1]
    
    test_ID = test_ID[:nrows]
    
else:    
    X=np.load('../data/nxtest_standard_original0.npy')
    X1= np.load('../data/nxtest_standard_derived0.npy')
    X2=pickle.load(open("../data/time_series_derived_test2.dat","rb"))
    X3=pickle.load(open("../data/time_series_original_test2.dat","rb"))
    X4=pickle.load(open("../data/cat_numeric_th60_test2.dat","rb"))
    xtest=np.hstack((X,X1,X2,X3,X4))
    del X,X1,X2,X3,X4
    
    
    if SelectedFeature:
        with open('../data/XGB_335Features Oct172015_044255_AUC_0p76115.p', 'rb') as fid:
            xgb_goodfeat = pickle.load(fid)
    
        good_features=list(xgb_goodfeat)
        xtest = xtest[:,good_features]

# make final predictions

ypreds_rf = RF.predict_proba(xtest)[:,1]


# In[97]:

with open('rf_ypreds_allfeat_'+outputfile+'.pkl', 'wb') as fid:
    pickle.dump(ypreds_rf, fid,protocol = 2)


# In[98]:

# generate submission files

def save_results(test_ID, predictions, filename):
    """Given a vector of predictions, save results in CSV format."""
    with open(filename, 'w') as f:
        f.write("ID,target\n")
        for i in range(len(test_ID)):
            f.write("%d,%f\n" % (test_ID[i], predictions[i]))
    
save_results(test_ID, ypreds_rf, 'rf_ypreds_allfeat_'+outputfile+'.csv')

