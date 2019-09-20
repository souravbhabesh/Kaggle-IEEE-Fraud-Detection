# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

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

df_full = df_full[1000:]
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline


# Custom Transformer that extracts columns passed as argument to its constructor
# BaseEstimator and TransformerMixin are base classes which are inherited
# we need to do is write our fit and transform methods and we get fit_transform for free.
class FeatureSelector(BaseEstimator, TransformerMixin):
    # Class Constructor
    def __init__(self, feature_names):
        self._feature_names = feature_names

    # Return self nothing else to do here
    def fit(self, X, y=None):
        return self

    # Method that describes what we need this transformer to do
    def transform(self, X, y=None):
        return X[self._feature_names]


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
            X = X.fillna(0)
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


# Numerical features to pass down the numerical pipeline
numerical_features = ['C1', 'C2', 'C6', 'C11', 'C13', 'C14']
null_list2 = ['D4', 'D6', 'D12', 'D14']

# Defining the steps in the numerical pipeline
numerical_pipeline1 = Pipeline(steps=[('num_selector', FeatureSelector(numerical_features)),
                                      ('imputer', SimpleImputer(strategy='median'))])

numerical_pipeline2 = Pipeline(steps=[('num_selector', FeatureSelector(null_list2)),
                                      ('imputer', SimpleImputer(strategy='median'))])

# Combining numerical and categorical pipeline into one full big pipeline horizontally
# using FeatureUnion
full_pipeline = FeatureUnion(transformer_list=[('numerical_pipeline1', numerical_pipeline1),
                                               ('numerical_pipeline2', numerical_pipeline2)])

# Building the model using the feature pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# Leave it as a dataframe becuase our pipeline is called on a
# pandas dataframe to extract the appropriate columns, remember?
X = df_full.drop('isFraud', axis=1)
# You can covert the target variable to numpy
y = df_full['isFraud'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# The full pipeline as a step in another pipeline with an estimator as the final step
rf = RandomForestClassifier(min_samples_split=500,
                            min_samples_leaf=50,
                            max_depth=8,
                            max_features='sqrt',
                            random_state=10)
full_pipeline_m = Pipeline(steps=[('full_pipeline', full_pipeline), ('model', rf)])

# Can call fit on it just like any other pipeline
full_pipeline_m.fit(X_train, y_train)

# Can predict with it like any other pipeline
y_pred = full_pipeline_m.predict(X_test)
print(X_test.shape)
print(y_pred.shape)
# print("Modeling complete")
pipe = Pipeline(steps=[('full_pipeline', full_pipeline),
                       ('model', rf)])
# Grid search for n_estimators
param_grid = {
    'model__n_estimators': [20, 30]
}

gsearch = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    # scoring='roc_auc',
    # n_jobs=-1,
    # iid=False,
    verbose=1000,
    cv=3)

print("Grid Search started")
# print(X.head())
gsearch.fit(X_train, y_train)

print("cv_results_:")
df_cv = pd.DataFrame.from_dict(gsearch.cv_results_)
# print(pd.DataFrame.from_dict(gsearch1.cv_results_))

print("best_params_:")
# print(gsearch.best_params_)


# print(gsearch.best_score_)
