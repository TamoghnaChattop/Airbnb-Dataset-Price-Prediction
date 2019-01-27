# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 14:57:45 2018

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
from sklearn.model_selection import train_test_split
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

result_train = np.zeros([5,4])
result_training = np.zeros([5,4])
result_test = np.zeros([5,4])

df_d, df_test, y_d, y_test = datasplit(df, y, test_size=0.20) 
df_train, df_val, y_train, y_val = datasplit(df_d, y_d, test_size=0.20)

#Linear regression based feature selection
estimator = LinearRegression()
selector_lin = rfecv(estimator, cv=10)
selector_lin.fit(df_train, y_train)
linsup = selector_lin.support_
linrank = selector_lin.ranking_

#Linear Regression
data_train = selector_lin.transform(df_train)
data_val = selector_lin.transform(df_val)
data_test = selector_lin.transform(df_test)

model = LinearRegression()
model.fit(data_train, y_train)
y_val_pred = model.predict(data_val)
print('Linear Regression : %f' % r2(y_val,y_val_pred))
print('Linear Regression (MSE) : %f' % mse(y_val,y_val_pred))
print('Linear Regression (Mean Absolute Error) : %f' % mabe(y_val, y_val_pred))
print('Linear Regression (Median Absolute Error) : %f' % mdae(y_val, y_val_pred))
result_train[1,0] = r2(y_val,y_val_pred)
result_train[1,1] = mse(y_val,y_val_pred)
result_train[1,2] = mabe(y_val, y_val_pred)
result_train[1,3] = mdae(y_val, y_val_pred)

#Training Scores
y_train_pred = model.predict(data_train)
result_training[1,0] = r2(y_train,y_train_pred)
result_training[1,1] = mse(y_train,y_train_pred)
result_training[1,2] = mabe(y_train,y_train_pred)
result_training[1,3] = mdae(y_train,y_train_pred)

#Testing set
y_test_pred = model.predict(data_test)
result_test[1,0] = r2(y_test,y_test_pred)
result_test[1,1] = mse(y_test,y_test_pred)
result_test[1,2] = mabe(y_test,y_test_pred)
result_test[1,3] = mdae(y_test,y_test_pred)

#RF based feature selection
estimator = RandomForestRegressor()
selector_rf = rfecv(estimator, cv=10)
selector_rf.fit(df_train, y_train)
rfsup = selector_rf.support_
rfrank = selector_rf.ranking_

#Random Forest Regressor
data_train = selector_rf.transform(df_train)
data_val = selector_rf.transform(df_val)
data_test = selector_rf.transform(df_test)

model = RandomForestRegressor()
model.fit(data_train, y_train)
y_val_pred = model.predict(data_val)
print('Random Forest Regressor : %f' % r2(y_val,y_val_pred))
print('Random Forest Regressor (MSE) : %f' % mse(y_val,y_val_pred))
print('Random Forest Regressor (Mean Absolute Error) : %f' % mabe(y_val, y_val_pred))
print('Random Forest Regressor (Median Absolute Error) : %f' % mdae(y_val, y_val_pred))
result_train[3,0] = r2(y_val,y_val_pred)
result_train[3,1] = mse(y_val,y_val_pred)
result_train[3,2] = mabe(y_val, y_val_pred)
result_train[3,3] = mdae(y_val, y_val_pred)

#Training Scores
y_train_pred = model.predict(data_train)
result_training[3,0] = r2(y_train,y_train_pred)
result_training[3,1] = mse(y_train,y_train_pred)
result_training[3,2] = mabe(y_train,y_train_pred)
result_training[3,3] = mdae(y_train,y_train_pred)

#Testing set
y_test_pred = model.predict(data_test)
result_test[3,0] = r2(y_test,y_test_pred)
result_test[3,1] = mse(y_test,y_test_pred)
result_test[3,2] = mabe(y_test,y_test_pred)
result_test[3,3] = mdae(y_test,y_test_pred)
