# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 23:16:32 2018

@author: acer
"""

import numpy as np
import pandas as pd

df= pd.read_excel("C:\\Users\\acer\\Desktop\\Book1.xlsx")

idx = df.groupby(['N1', 'N2'])['N3'].transform(max) == df['N3']

# use the index to fetch correct rows in dataframe
print(df[idx])
