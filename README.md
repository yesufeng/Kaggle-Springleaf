# Kaggle-Springleaf
Kaggle competition: Springleaf Marketing Response by team CCTO

## Introduction
This repo contains ipython notebooks prepared for the Kaggle competition: Springleaf Marketing Response.Springleaf offers their customers personal and auto loans that help them take control of their lives and their finances. Direct mail is one important way Springleaf's team can connect with customers whom may be in need of a loan. In order to improve their targeted effort, Springleaf would like to focus on the customers who are likely to respond and be good candidates for their services.

Using a large set of <em>anonymized</em> features, Springleaf is asking us to predict which customers will respond to a direct mail offer. 

## Data
A high-dimensional dataset of <em>anonymized</em> customer information is provided. Each entry (row) corresponds to one customer. the response variable is binary. There are over 140,000 entries in both the test and training set. 

## Project Pipeline
### Data preprocessing and feature extraction
In the preprocessing folder, feature data were processed differently based on different data types. 
  1. Numerical data was preprocessed in data_preprocessing_SL_Oct20_train_test_th60.ipynb. Key processing include missing value imputation, outlier detection, log-transform of right-skewed columns, standardization of numerical columns, etc. Besides basic numerical columns, 10 numerical columns were derived. Categorical columns with limited number of values were transformed using DictVectorizer (OneHot encoding). Numerical columns with too few values are separated from other numerical columns, so as the time series columns.
  2. Time series data were processed in data_preprocessing_SL_oct20_time_series_normalization.ipynb
  3. Categorical columns with too many values as well as numerical columns with too few values were processed in data_preprocessing_SL_oct20_cat_num_normalization.ipynb
  4. All other categorical columns were preprocessed using OneHot encoding in data_preprocessing_SL_oct20_th60_cat_label_encoding.ipynb

### Feature selection
Feature selection were done with notebooks in the feature_selection folder. Multiple methods were used to do feature selection, including RFECV, greedy forward selection, backward selection and the SelectKBest from sklearn.
Model inputs for different models:
  1. linear models (Logistic, SVM, Passive aggressive): numerical variables
  2. tree-based models (xgBoost, random forest, scikit learn gradient boosting): numerical + catigorical variables

### Grid search and model optimization
Grid search for hyperparameter tuning was done using either sklearn gridsearchCV or the home-built method that generates prediction on the test set during cross-validation, the prediction can be used later as meta-features. Grid search were done with different algorithms such as xgboost, random forest, online svm and logistic regression.

### Final prediction
Final predictions are made with both level-0 and level-1 models using both basic features, derived features, and meta-features, using models including xgBoost, RandomForest, SGD-logistic regression, SGD support vector machines, SDG passive agressive classfier.

### Model ensembling
Model ensembling using greedy bagging method: http://www.cs.cornell.edu/~caruana/ctp/ct.papers/caruana.icml04.icdm06long.pdf

The idea is to select the models from a library of models to achieve the highest cross-validation score (AUC in this case).
The models are selected using a forward greedy approach with replacement, i.e., at each step a new model is added to the existing model bag to optimize the score. At the end of the selection, the weight of each model is given by the number of times it appears in the bag, and is used to perform the final ensembling on the test predictions.

