# -*- coding: utf-8 -*-
"""pandas.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/19JJH8e7Pe_WOcmcxOoOf8DSSwIS-jz4D
"""

import numpy as np
import pandas as pd

df = pd.read_csv("diabetes.csv")
df

df.head(5)

df.groupby(['Pregnancies']).median()

df.tail(5)

df.shape

df.dtypes

# statistics criteria
  x = df.describe()
  x

# See column names
 col_names = df.columns
 col_names

# See row names
y = list(x.index)
y

# Subset rows and columns by index number
 df.iloc[4:10,2:5]

# Select a column by name
df['BMI']
df.BMI

# Select columns by names
df[['Glucose','Insulin','Outcome']]

x.loc["mean","BMI"]

a = df.loc[df.BMI < 20,["Glucose","Insulin",'Outcome']]
 a.head(5)

a.reset_index(drop=True, inplace=True)

a.head(5)

len(a)

a["Glucose"].mean()

print(sum(a.Outcome)/len(a))

b = df.loc[df.BMI > 25,["Glucose","Insulin",'Outcome']]
b.shape

b["Glucose"].mean()

print(sum(b.Outcome)/len(b))

df2 = pd.read_csv("mice_pheno.csv")
df2.tail(20)

x = df2.loc[df2['Sex'] == "M"]
x.head(10)

x = df2.loc[(df2['Sex'] == "M") & (df2['Diet'] =="hf")]

df2['Diet'].value_counts()

df2.groupby(by = ['Sex','Diet']).size()

df2.head(5)

df2.loc[df2['Sex'] == 'M','Sex'] = 1
df2.loc[df2['Sex'] == 'F','Sex'] = 0
df2.head(5)
df2.tail(10)

df2.groupby(['Sex']).size()

df.groupby(['Age']).median()

df.groupby(['Age']).median().sort_values('Glucose')

df.groupby(['Pregnancies']).mean()

# Make a new column
df['Total'] = df['Glucose']+ df['Insulin']
df.head(5)

df = df.drop(columns = 'Total' )
df.head(5)

df.to_csv("df2.csv",index = False)

df['Total1'] = df['Glucose']+ df['Insulin']
dfdf['Total2'] = df['Glucose']+ df['Insulin']



s2 = df.filter(regex= 'e$|^B')
s2.head(10)
