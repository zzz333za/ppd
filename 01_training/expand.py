# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 03:02:02 2019

@author: 12583
"""


import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from lightgbm.sklearn import LGBMClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse
from scipy.stats import kurtosis
import time
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

train_df = pd.read_csv('../data/train.csv', parse_dates=['auditing_date', 'due_date', 'repay_date'])

l0=[]
l1=[]
l2=[]
l3=[]
l4=[]
c=0
for j in range(train_df.shape[0]):
    ui=train_df['user_id'].values[j]
    li=train_df['listing_id'].values[j]
    ad=train_df['auditing_date'].values[j]
    due=train_df['due_date'].values[j]
    out=0
    now=ad
    while out==0:
        l0.append(ui)
        l1.append(li)
        l2.append(ad)
        l3.append(due)
        l4.append(now)
        now=now+np.timedelta64(1 ,'D')
        if now>due+np.timedelta64(1 ,'D'):
            out=1
            c=c+1
    if c%10000==0:
        print(c,'/',train_df.shape[0])
data=pd.DataFrame()
data['user_id']=l0
data['listing_id']=l1
data['auditing_date']=l2
data['due_date']=l3
data['repay_date']=l4
data.to_csv('newlist.csv',index=False)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        