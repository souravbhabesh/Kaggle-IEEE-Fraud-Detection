# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from pprint import pprint
from sklearn.preprocessing import OneHotEncoder

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

# df_full = df_full[:1000]

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline


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


# Numerical features to pass down the numerical pipeline, replace missing with 0
numerical_features = ['TransactionAmt', 'C1', 'C2', 'C6', 'C11', 'C13', 'C14']
# Numeric columns which have NULL values replaced with -200
null_list2 = ['D4', 'D6', 'D12', 'D14']
# Numeric columns which have NULL values replaced with -1
null_list1 = ['dist1', 'dist2', 'D1', 'D2', 'D7', 'D8', 'D9']

# Categorical feature list
# categorical variables replace missing with None
cat_list1 = ['ProductCD', 'card4', 'P_emaildomain', 'R_emaildomain',
             'id_12', 'id_15', 'id_16', 'id_23', 'id_27', 'id_28',
             'id_29', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38',
             'DeviceType']
# Categorical features which require some factor levels to be combined
cat_list2 = ['card6']
cat_list = ['ProductCD', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'card4', ]
print("Full predictor list: ")
print(numerical_features+null_list1+null_list2+cat_list1)


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

# Combining numerical and categorical pipeline into one full big pipeline horizontally
# using FeatureUnion
full_pipeline = FeatureUnion(transformer_list=[('numerical_pipeline1', numerical_pipeline1),
                                               ('numerical_pipeline2', numerical_pipeline2),
                                               ('numerical_pipeline3', numerical_pipeline3),
                                               ('categorical_pipeline', categorical_pipeline)])

# Building the model using the feature pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# Leave it as a dataframe becuase our pipeline is called on a
# pandas dataframe to extract the appropriate columns, remember?
X = df_full.drop('isFraud', axis=1)
# You can covert the target variable to numpy
y = df_full['isFraud'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# Use combined features to transform dataset:
X_features = full_pipeline.fit(X_train, y_train).transform(X_train)
print("Shape of combined space ", X_features.shape, "features")
print("Combined space has", X_features.shape[1], "features")
# print(X_features[:,10])

from sklearn.svm import SVC

svm = SVC(kernel="poly", probability=True)

# Do grid search over k, n_components and C:

pipeline = Pipeline([("features", full_pipeline), ("svm", svm)])
#
param_grid = dict(  # features__pca__n_components=[1, 2, 3],
    # features__univ_select__k=[1, 2],
    svm__C=[0.1, 1, 10])

grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=5, verbose=10)
print("Grid search started")
# grid_search.fit(X_train, y_train)


# The full pipeline as a step in another pipeline with an estimator as the final step
rf = RandomForestClassifier(min_samples_split=500,
                            min_samples_leaf=50,
                            max_depth=8,
                            max_features="sqrt",
                            class_weight='balanced',
                            random_state=10)
# full_pipeline_m = Pipeline(steps=[('full_pipeline', full_pipeline), ('model', rf)])

# Can call fit on it just like any other pipeline
# full_pipeline_m.fit(X_train, y_train)

# Can predict with it like any other pipeline
# y_pred = full_pipeline_m.predict(X_test)
# print(X_test.shape)
# print(y_pred.shape)
# print("Modeling complete")
pipe = Pipeline(steps=[('full_pipeline', full_pipeline),
                       ('model', rf)])
# Grid search for n_estimators
param_grid = {
    'model__n_estimators': range(550, 600, 50)
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
# gsearch.fit(X_train, y_train)

print("cv_results_:")
# df_cv = pd.DataFrame.from_dict(gsearch.cv_results_)
# pprint(pd.DataFrame.from_dict(gsearch.cv_results_))

print("best_params_:")
# print(gsearch.best_params_)

print("****************** Predicting******************")

# rf_final = gsearch.best_estimator_
# final_pipeline = Pipeline(steps=[('full_pipeline', full_pipeline),
#                                  ('model', rf_final)])
#
# # Can call fit on it just like any other pipeline
# final_pipeline.fit(X_train, y_train)

# y_pred = gsearch.predict_proba(X_test)

from sklearn.metrics import roc_auc_score

# print("AUC for test set: ", roc_auc_score(y_test, y_pred[:, 1]))

#
rf_final = RandomForestClassifier(n_estimators=550,
                                  min_samples_split=500,
                                  min_samples_leaf=50,
                                  max_depth=8,
                                  max_features="sqrt",
                                  class_weight='balanced',
                                  random_state=10)
final_pipeline = Pipeline(steps=[('full_pipeline', full_pipeline),
                                 ('model', rf_final)])

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
