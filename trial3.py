# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 16:03:10 2018

@author: tchat
"""

import csv
import numpy as np 
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.utils import shuffle
import math
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn import metrics
from sklearn.base import TransformerMixin
from sklearn import tree
from sklearn.feature_selection import RFECV as rfecv
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mabe
from sklearn.metrics import median_absolute_error as mdae
from sklearn.metrics import r2_score as r2
from sklearn.model_selection import train_test_split as datasplit
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFECV as rfecv

df = pd.read_csv('D:\EE 660 Project AIRBNB\Exploratory.csv')

df['security_deposit']=df['security_deposit'].str.replace('$', '')
df['security_deposit']=df['security_deposit'].str.replace(',', '')
df['security_deposit'] = df['security_deposit'].astype('float64') 

df['cleaning_fee']=df['cleaning_fee'].str.replace('$', '')
df['cleaning_fee']=df['cleaning_fee'].str.replace(',', '')
df['cleaning_fee'] = df['cleaning_fee'].astype('float64') 

df['extra_people']=df['extra_people'].str.replace('$', '')
df['extra_people']=df['extra_people'].str.replace(',', '')
df['extra_people'] = df['extra_people'].astype('float64') 


class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value 
        in column.

        Columns of other types are imputed with mean of column.

        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)

X = pd.DataFrame(df)
df = DataFrameImputer().fit_transform(X)



x=(df.groupby('neighbourhood_cleansed')['price'].sum())
y=(df['neighbourhood_cleansed'].value_counts(sort=False))
#print(x)
#print(y)

#x.to_csv('/Users/Penguin/Desktop/x.csv')
#y.to_csv('/Users/Penguin/Desktop/y.csv')

df['host_is_superhost'] = np.where(df['host_is_superhost'] == 't',1,0)
df['instant_bookable'] = np.where(df['instant_bookable'] == 't',1,0)
df['require_guest_profile_picture'] = np.where(df['require_guest_profile_picture'] == 't',1,0)
df['require_guest_phone_verification'] = np.where(df['require_guest_phone_verification'] == 't',1,0)
df['host_response_rate'] = df['host_response_rate'].str.replace("%", "").astype("float")

df = pd.get_dummies(df, columns = ['property_type'])
df = pd.get_dummies(df, columns = ['neighbourhood_cleansed'])
df = pd.get_dummies(df, columns = ['cancellation_policy'])

cleanup_data = {"bed_type":     {"Real Bed": 5, "Futon": 4, "Pull-out Sofa":3, "Airbed":1,"Couch":2},
                "room_type": {"Entire home/apt": 3, "Private room": 2, "Shared room": 1}
                }

df.replace(cleanup_data, inplace=True)


clean = {"host_response_time": {"a few days or more":4,"within a day":3,"within a few hours":2,"within an hour":1}}
df.replace(clean,inplace=True)

#df.to_csv(path_or_buf='/Users/Penguin/Desktop/Exp-Processed.csv')

count = df['amenities'].str.split(",").apply(len)
df['amenities']=count

df['zipcode'] = df['zipcode'].str[:5]

y=df['price']
df=df.drop('price',1)
df=df.drop('host_since',1)
df=df.drop('calendar_last_scraped',1)
df=df.drop('neighbourhood',1)
df=df.drop('host_verifications',1)
df=df.drop('zipcode',1)
df=df.drop('host_identity_verified',1)

tree_depth = np.zeros([18,1])

for val in range(2,20):
    df_d, df_test, y_d, y_test = datasplit(df, y, test_size=0.20) 
    model = tree.DecisionTreeRegressor(max_depth=val)
    tree_depth[val-2,0] = np.mean(cross_val_score(model,df_d,y_d,cv=5))
    
depthval = np.argmax(tree_depth)+2
       
print('Depth Value : %f' % depthval)

sample_split = np.zeros([18,1])

for val in range(2,20):
    df_d, df_test, y_d, y_test = datasplit(df, y, test_size=0.20) 
    model = tree.DecisionTreeRegressor(min_samples_split=val)
    sample_split[val-2,0] = np.mean(cross_val_score(model,df_d,y_d,cv=5))
    
s_split = np.argmax(sample_split)+2   

print('Samples Split : %f' % s_split) 
