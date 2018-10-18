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
y= data["income"]

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


cat_list,num_list=variable_list(X)

# Making dataframes of categorical variables and continous variables
def df_split(df):
    df_cat=pd.DataFrame()
    df_num=pd.DataFrame()
    for col_name in df.columns:
        if df[col_name].dtypes=="object":
            df_temp= df[col_name]
            df_cat= pd.concat([df_cat,df_temp],axis=1)
        else:
            df_temp_1= df[col_name]
            df_num= pd.concat([df_num,df_temp_1],axis=1)
    return df_cat,df_num

X_cat,X_num= df_split(X)


# Missing Value Function
def missing(df):
    df1=pd.DataFrame()
    for x in df.columns:
        N = len(df[x])
        NMiss = df[x].isnull().sum()
        Miss_prop = (NMiss/N)*100
        df2 = pd.DataFrame([N,NMiss,Miss_prop], index=['N','Miss','Miss_Prop'])
        df2.columns= [df[x].name]
        df1 = pd.concat([df1,df2],axis=1)
    df1.reset_index(inplace= True)
    return df1

# Statistical report of variables

# For Numeric Variables    
 def summary_stats_num(x):
     N = len(x)
     Nmiss = x.isnull().sum()
     Min = np.min(x)
     Q1 = np.nanpercentile(x,25)
     Q3 = np.nanpercentile(x,75)
     Mean = np.nanmean(x)
     Median = np.nanmedian(x)
     Q90 = np.nanpercentile(x,90)
     Q95 = np.nanpercentile(x,95)
     Q99 = np.nanpercentile(x,99)
     Max = np.max(x)
     IQR = Q3-Q1
     SD = np.nanstd(x)
     df = pd.DataFrame([N,Nmiss,Min,Q1,Q3,Mean,Median,Q90,Q95,Q99,Max,IQR,StdDev])
     df.columns = [x.name]
     return df

 def summary_num(df,num_list):
     df1 = pd.DataFrame()
     for x in num_list:
         d= summary_stats_num(df[x])
         df1 = pd.concat([df1,d],axis=1)
     df1.reset_index(inplace=True)
     df1.rename(columns={df1.columns[0]:'Statistic'},inplace=True)
     return df1


# For Categorical Variables 
 def summary_stats_cat(X):
     x = pd.DataFrame(X.value_counts())
     x.reset_index(inplace=True)
     x.columns = [X.name,'Count']
     return x
 
def summary_cat(df,cat_list):
    df1 = pd.DataFrame()
    for x in cat_list:
        d = summary_stats_cat(df[x])
        df1 = pd.concat([df1,d],axis=1)
    return df1


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


X_newer= con_impute(strategy='mean').fit_transform(X_new)


# Outlier Treatment

# Outlier Detection using Tukey IQR method (Values below Q1 - 1.5*IQR and Values above Q3+ 1.5IQR)
def detect_outlier(x):
    q1 = np.nanpercentile(x,25)
    q3 = np.nanpercentile(x,75)
    iqr = q3-q1
    floor = q1-1.5*iqr
    ceiling = q3+1.5*iqr
    outlier_indices = list(x.index[(x<floor)|(x>ceiling)])
    outlier_values = list(x[outlier_indices])
    return outlier_indices,outlier_values

# Outlier Treatment
def outlier_removal(df,num_list):
    for x in num_list:
        q1 = np.nanpercentie(df[x],25)
        q3 = np.nanpercentile(df[x],75)
        iqr = q3-q1
        floor = q1-1.5*iqr
        ceiling = q3+1.5*iqr
        index = list(df[x].index[(df[x]>floor)|(df[x])<ceiling])
        df[x] = [np.nanpercentile(df[x],95) for df[x].index in index]
    return df


# Descriptive Data Visualization


# For Categorical Variables
def plot_cat(df):
    fig=plt.figure(figsize=(10,30))
    for c,num in zip(df.columns, range(1,len(df.columns)+1)):
        ax = fig.add_subplot(len(df.columns),1,num)
        ax.plot=(df[c].value_counts().plot(kind='bar'))
        ax.set_title(c)

    plt.tight_layout()
    plt.show()
    #return fig
    
plot_cat(X_cat)


# For Continous Variables   
fig= plt.figure(figsize=(10,30))
for c,num in zip(X_num.columns, range(1,len(X_num.columns)+1)):
    ax1= fig.add_subplot(len(X_num.columns),1,num)
    ax1.plot= sns.distplot(X_num[c],bins=20)
    #ax.set_title(c)
plt.tight_layout()
#plt.show()

fig1= plt.figure(figsize=(10,30))
for c,num in zip(X_num.columns, range(1,len(X_num.columns)+1)):
    ax2= fig1.add_subplot(len(X_num.columns),1,num)
    ax2.plot= sns.boxplot(X_num[c],orient='vertical')
    #ax.set_title(c)
plt.tight_layout()
#plt.show()


fig_2= plt.figure(figsize=(10,30))
for c,num in zip(X_num.columns, range(1,len(X_num.columns)*2)):
    if num%2!=0:
        ax3= fig_2.add_subplot(len(X_num.columns),2,num)
        ax3.plot= sns.distplot(X_num[c],bins=20)
    #ax.set_title(c)
plt.tight_layout()
#plt.show()

 
sns.distplot(X_num['age'],bins=20)
sns.boxplot(X_num['age'],orient='vertical')

fig= plt.figure()
axes=fig.add_subplot(2,2,2)
