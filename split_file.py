import pandas as pd

df_transaction_test = pd.read_csv('Data/test_transaction.csv')
df_identity_test = pd.read_csv('Data/test_identity.csv')
df_test = pd.merge(df_transaction_test, df_identity_test, left_on='TransactionID', right_on='TransactionID', how='left')

df_test.to_csv('Data/dftest.csv', index=False)