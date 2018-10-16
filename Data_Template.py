# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 23:22:28 2018

@author: acer
"""

#Importing Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import TransformerMixin


#Importing Dataset

data= pd.read_csv("C:\\Users\\acer\\Desktop\\adult\\adult.csv")

data.head()

#Checking data types of each column
data.info()


#Changing data type of dependent variable (income in this dataset) into binary (0 if income<= 50k else 1)
data["income"]=[0 if x=="<=50K" else 1 for x in data["income"]]

#Descriptive visualization for variables

sns.pairplot(data)

#Making dataframe and list of independent variables and dependent variable

X= data.drop("income",axis=1)
Y= data["income"]

# Checking missing data
X.isnull().sum().sort_values(ascending=False)


# Check Later
for col_name in X.columns:
    if X[col_name].dtypes=="object":
        unique_cat= len(X[col_name].unique)
        print("Feature '{col_name}' has {unique_cat} unique categories".format(col_name=col_name,unique_cat=unique_cat))

#Creating automatic categorical and continous variable name lists

def variable_list(df):
    cat_list=[]
    con_list=[]
    for col_name in df.columns:
        if df[col_name].dtypes=="object":
            cat_list.append(df[col_name].name)
        else:
            con_list.append(df[col_name].name)
    return cat_list,con_list


c_list,co_list=variable_list(X)

# Statistical report of variables
 def summary_stats_cont(x):
     Nmiss = x.isnull().sum()
     Min = np.min(x)
     Q1 = np.percentile(x,25)
     Q3 = np.percentile(x,75)
     Mean = np.mean(x)
     Median = np.median(x)
     Q95 = np.percentile(x,95)
     Q99 = np.percentile(x,99)
     Max = np.max(x)
     IQR = Q3-Q1
     SD = np.std(x)
     
     return [Nmiss,Min,Q1,Q3,Mean,Median,Q95,Q99,Max,IQR,SD]
 
 def summary_stats_cat(X):
     x=pd.DataFrame(X.value_counts())
     x.reset_index(inplace=True)
     return x
 
sta= summary_stats_cat(X['workclass'])     
    
stats=summary_stats_cont(X['age'])

x=X['workclass'].value_counts()
x= pd.DataFrame(X['workclass'].value_counts())
x.reset_index(inplace=True)






# Imputation Functions
        
# For categorical variables

class SeriesImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        If the Series is of dtype Object, then impute with the most frequent object.
        If the Series is not of dtype Object, then impute with the mean.  

        """
    def fit(self, X, y=None):
        if   X.dtype == np.dtype('O'): self.fill = X.value_counts().index[0]
        else                            : self.fill = X.mean()
        return self

    def transform(self, X, y=None):
       return X.fillna(self.fill)

def cat_impute(df,cat_list):
    for x in cat_list:
        temp=df[x]
        imp= SeriesImputer()
        imp.fit(temp)
        dummy= imp.transform(temp)
        df=df.drop(x,1)
        df= pd.concat([df,dummy],axis=1)
    return df
        

class con_impute(TransformerMixin):
    def __init__(self, strategy='mean',filler='NA'):
       self.strategy = strategy
       self.fill = filler

    def fit(self, X, y=None):
       if self.strategy == 'mean':
           self.fill = X.mean()
       elif self.strategy == 'median':
           self.fill = X.median()
       elif self.strategy == 'mode':
           self.fill = X.mode().iloc[0]
       return self
    def transform(self, X, y=None):
       return X.fillna(self.fill)


X_new= cat_impute(X,cat_list)

X_new.isnull().sum().sort_values(ascending=False)

X_newer= con_impute(strategy='mean').fit_transform(X_new)

X_newer.isnull().sum().sort_values(ascending=False)















