#!/opt/homebrew/bin/python3

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

#filter data for relevant information
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
df_churn = pd.read_csv('telco_churn.csv')
df_churn = df_churn[['gender', 'PaymentMethod', 'MonthlyCharges', 'tenure', 'Churn']].copy()
print(df_churn.head())

#copy data frame to a new variable and replace missing values with zero
df = df_churn.copy()
df.fillna(0, inplace=True)

encode = ['gender','PaymentMethod']

for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]

    import numpy as np 

df['Churn'] = np.where(df['Churn']=='Yes', 1, 0)
#Now, let’s define our input and output :
X = df.drop('Churn', axis=1)
Y = df['Churn']

#fit model using random forest
clf = RandomForestClassifier()
clf.fit(X, Y)

#save model
pickle.dump(clf, open('churn_clf.pkl', 'wb'))