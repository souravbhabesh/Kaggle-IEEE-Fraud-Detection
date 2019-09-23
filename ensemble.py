import pandas as pd

df_xgb = pd.read_csv("Data/Submission/XGBoost/submission.csv")
df_lgbm = pd.read_csv("Data/Submission/LGBM/submission.csv")

a = 0.65
b = 0.35

df = df_xgb.copy()
df.columns = [['TransactionID','isFraud_xgb']]
df['isFraud'] = (a * df_lgbm['isFraud'] + b * df_xgb['isFraud'])

df = df.drop(columns=['isFraud_xgb'])

print(df.describe())

df.to_csv('Data/Submission/Ensemble/submission.csv', index=False)