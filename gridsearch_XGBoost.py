# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from pprint import pprint
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import Normalizer
from sklearn.metrics import roc_auc_score

# Building the model using the feature pipeline
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 1000)
from pprint import pprint
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

for dirname, _, filenames in os.walk('Data'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# %% [code]
# Read in the transaction and identity data
# Merge the two dataframes using TransactionID as key
df_transaction = pd.read_csv('Data/train_transaction.csv')
df_identity = pd.read_csv('Data/train_identity.csv')

df_full = pd.merge(df_transaction, df_identity, left_on='TransactionID', right_on='TransactionID', how='left')

# df_full = df_full[:10000]

# Train Test split
X = df_full.drop('isFraud', axis=1)
# You can covert the target variable to numpy
y = df_full['isFraud'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# Utility functions
def percentageFraud(df_train, df_test, col, target='isFraud'):
    table_train = pd.crosstab(df_train[col], df_train[target])
    table_train['%fraud'] = (table_train[1] / (table_train[0] + table_train[1])) * 100
    table_train = table_train.sort_values(by='%fraud', ascending=False)
    table_train['index'] = table_train.index
    factor_dict = {}
    # List of factor levels where %fraud > 0
    key_list = list(table_train[table_train['%fraud'] > 0]['index'])
    for key in key_list:
        value = key
        factor_dict[key] = value
    # List of factor levels where %fraud == 0
    factor_levels = set(table_train['index'])
    for key in key_list:
        factor_levels.discard(key)
    for key in factor_levels:
        factor_dict[key] = 'other'
    pprint(factor_dict)
    return factor_dict


# Custom Transformer that extracts columns passed as argument to its constructor
# BaseEstimator and TransformerMixin are base classes which are inherited
# we need to do is write our fit and transform methods and we get fit_transform for free.
class FeatureSelector(BaseEstimator, TransformerMixin):
    # Class Constructor
    def __init__(self, feature_names):  # paramter name and self.name should be the same
        self.feature_names = feature_names

    # Return self nothing else to do here
    def fit(self, X, y=None):
        return self

    # Method that describes what we need this transformer to do
    def transform(self, X, y=None):
        # print("Feature Selector")
        return X[self.feature_names].to_numpy()


# Custom transformer for Numerical variables
class NumericalTransformer(BaseEstimator, TransformerMixin):
    # Class Constructor
    def __init__(self, impute_C=True):
        self._impute_C = impute_C
        # self._C2 = C2

    # Return self, nothing else to do here
    def fit(self, X, y=None):
        return self

    # Custom transform method
    def transform(self, X, y=None):
        # Check if needed
        if self._impute_C:
            # Impute columns matching C*
            X = X.fillna(0)  # this doesn't work
        #     # create new column
        #     X.loc[:, 'bath_per_bed'] = X['bathrooms'] / X['bedrooms']
        #     # drop redundant column
        #     X.drop('bathrooms', axis=1)
        # # Check if needed
        # if self._years_old:
        #     # create new column
        #     X.loc[:, 'years_old'] = 2019 - X['yr_built']
        #     # drop redundant column
        #     X.drop('yr_built', axis=1)
        #
        # # Converting any infinity values in the dataset to Nan
        # X = X.replace([np.inf, -np.inf], np.nan)
        # returns a numpy array
        return X.values


# Custom transformer that breaks dates column into year, month and day into separate columns and
# converts certain features to binary
class CategoricalTransformer(BaseEstimator, TransformerMixin):
    # Class constructor method that takes in a list of values as its argument
    def __init__(self, factor_dict):
        self.factor_dict = factor_dict

    # Return self nothing else to do here
    def fit(self, X, y=None):
        return self

    # Transformer method we wrote for this transformer
    def transform(self, X, y=None):
        X.loc[:, 'P_emaildomain'] = X['P_emaildomain'].map(self.factor_dict)
        # returns numpy array
        return X.values


# Numerical features to pass down the numerical pipeline, replace missing with 0
numerical_features = ['TransactionAmt', 'C1', 'C2', 'C6', 'C11', 'C13', 'C14']
# Numeric columns which have NULL values replaced with -200
null_list2 = ['D4', 'D6', 'D12', 'D14']
# Numeric columns which have NULL values replaced with -1
null_list1 = ['dist1', 'dist2', 'D1', 'D2', 'D7', 'D8', 'D9']
# PCA list
v_list = []
for col in df_full:
    if 'V' in col:
        # print(col)
        v_list.append(col)

# Categorical feature list
# categorical variables replace missing with None
cat_list1 = ['ProductCD', 'card4',
             'id_12', 'id_15', 'id_16', 'id_23', 'id_27', 'id_28',
             'id_29', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38',
             'DeviceType']
# Categorical features which require some factor levels to be combined
cat_list3 = ['P_emaildomain']
cat_list2 = ['card6', 'P_emaildomain', 'R_emaildomain']
cat_list = ['ProductCD', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'card4', ]

print("Full predictor list: ")
print(numerical_features + null_list1 + null_list2 + cat_list1)

# Defining the steps in the numerical pipeline
numerical_pipeline1 = Pipeline(steps=[('num_selector', FeatureSelector(numerical_features)),
                                      ('imputer', SimpleImputer(strategy="constant", fill_value=0))])
# ('imputeC', NumericalTransformer(impute_C=True))])

numerical_pipeline2 = Pipeline(steps=[('num_selector', FeatureSelector(null_list2)),
                                      ('imputer', SimpleImputer(strategy="constant", fill_value=-200))])

numerical_pipeline3 = Pipeline(steps=[('num_selector', FeatureSelector(null_list1)),
                                      ('imputer', SimpleImputer(strategy="constant", fill_value=-1))])

categorical_pipeline = Pipeline(steps=[('cat_selector', FeatureSelector(cat_list1)),
                                       ('imputer', SimpleImputer(strategy="constant", fill_value=None)),
                                       ('one_hot_encoder', OneHotEncoder(sparse=False, drop='first'))])

cat_pipe = Pipeline(steps=[('cat_selector', FeatureSelector(cat_list3)),
'condense_factor', CategoricalTransformer(factor_dict=percentageFraud(df_train, df_test, col, target='isFraud')),
                           ('one_hot_encoder', OneHotEncoder(sparse=False, drop='first'))])

# This dataset is way too high-dimensional. Better do PCA:
pca = PCA(n_components=2)

# Maybe some original features where good, too?
selection = SelectKBest(k=1)

pca_pipeline = Pipeline(steps=[('pca_selector', FeatureSelector(v_list)),
                               ('imputer', SimpleImputer(strategy="constant", fill_value=-1)),
                               ('normalizer', Normalizer(copy=False)),
                               ('pca_kbest', FeatureUnion([("pca", pca), ("univ_select", selection)]))])

# Combining numerical and categorical pipeline into one full big pipeline horizontally
# using FeatureUnion
full_pipeline = FeatureUnion(transformer_list=[('numerical_pipeline1', numerical_pipeline1),
                                               ('numerical_pipeline2', numerical_pipeline2),
                                               ('numerical_pipeline3', numerical_pipeline3),
                                               ('pca_pipeline', pca_pipeline),
                                               ('condense_factor', cat_pipe),
                                               ('categorical_pipeline', categorical_pipeline)])

# Use combined features to transform dataset:
X_features = full_pipeline.fit(X_train, y_train).transform(X_train)
print("Shape of combined space ", X_features.shape, "features")
print("Combined space has", X_features.shape[1], "features")
# print(X_features[:,10])


# XGBoost model
params = {'learning_rate': 0.1,
          'n_estimators': 1000,
          'max_depth': 5,
          'min_child_weight': 1,
          'gamma': 0,
          'subsample': 0.8,
          'colsample_bytree': 0.8,
          'objective': 'binary:logistic',
          'nthread': 4,
          'scale_pos_weight': 1,
          'seed': 27}

xgb = XGBClassifier(**params)

pipe = Pipeline(steps=[('full_pipeline', full_pipeline),
                       ('model', xgb)])
# Grid search for n_estimators
param_grid = {
    'model__max_depth': range(3, 10, 2),
    'model__min_child_weight': range(1, 6, 2)
}

gsearch = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring='roc_auc',
    # n_jobs=2,
    iid=False,
    verbose=1000,
    cv=3)

print("Grid Search started")
# print(X.head())
gsearch.fit(X_train, y_train)

print("cv_results_:")
df_cv = pd.DataFrame.from_dict(gsearch.cv_results_)
pprint(pd.DataFrame.from_dict(gsearch.cv_results_))

print("best_params_:")
print(gsearch.best_params_)

print("****************** Predicting******************")

y_pred = gsearch.predict_proba(X_test)

print("AUC for test set: ", roc_auc_score(y_test, y_pred[:, 1]))

#
params.update(gsearch.best_params_)
xgb_final = XGBClassifier(**params)
final_pipeline = Pipeline(steps=[('full_pipeline', full_pipeline),
                                 ('model', xgb_final)])

# Can call fit on it just like any other pipeline
final_pipeline.fit(X_train, y_train)
print("AUC for test set: ", roc_auc_score(y_test, final_pipeline.predict_proba(X_test)[:, 1]))

print("******************** Generating the submission file******************")
df_transaction_test = pd.read_csv('Data/test_transaction.csv')
df_identity_test = pd.read_csv('Data/test_identity.csv')
df_test = pd.merge(df_transaction_test, df_identity_test, left_on='TransactionID', right_on='TransactionID', how='left')

y_pred = final_pipeline.predict_proba(df_test)

df_score = pd.DataFrame({'isFraud': y_pred[:, 1]})
df_score['TransactionID'] = df_test['TransactionID']
df_score = df_score[['TransactionID', 'isFraud']]

print(df_score.head())

df_score.to_csv('Data/Submission/submission.csv', index=False)
