# -*- coding: utf-8 -*-
"""Untitled59.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Agm2RPUpJlUfWLCf9ov5YqHHyiGBf2h7
"""

import pandas as pd
import pylab as pl
import numpy as np
import seaborn as sns
import scipy.optimize as opt
from sklearn import preprocessing
import matplotlib.pyplot as plt

data = pd.read_csv('telecom.csv')
data.head()

plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(),annot=True, fmt=".2f")

data = data.drop(['region','reside'], axis=1)
data.head()

data.custcat.value_counts()

x = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

from sklearn import preprocessing

x = preprocessing.StandardScaler().fit(x).transform(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LogisticRegression

LR = LogisticRegression().fit(x_train,y_train)

y_pred = LR.predict(x_test)
y_pred

y_Pred_prob = LR.predict_proba(x_test)
y_Pred_prob

"""check "https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.LogisticRegression.html

"""

from sklearn.metrics import accuracy_score

print("accuracy = ", accuracy_score(y_test,y_pred))

from sklearn.metrics import ConfusionMatrixDisplay

y = LogisticRegression().fit(x_train, y_train)
ConfusionMatrixDisplay.from_estimator(y, x_test, y_test)



data.head(10)

data.loc[data ['custcat'] == 2, ['custcat']] = 1
data.loc[data['custcat'] == 3, ['custcat']] = 1

data.head(10)