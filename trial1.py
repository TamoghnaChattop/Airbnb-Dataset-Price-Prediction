# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 19:50:09 2018

@author: tchat
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import statsmodels
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv('D:\EE 660 Project AIRBNB\Exploratory.csv')

#var = df['property_type'].value_counts()
#x = df.loc[df['property_type'] == 'Dome house', 'price'].sum()

var2 = df['zipcode'].value_counts()
var3 = df['host_response_time'].value_counts()

#df1 = pd.read_csv(r'D:\EE 660 Project AIRBNB\No. of listings per zipcode.csv')
#pl = df1.plot.bar(rot=0)
#fig = pl.get_figure()
#fig.savefig("No.of Listings per Zipcode.png")

#plt.scatter(df['price'],df['bedrooms'])
#plt.ylabel('bedrooms')
#plt.xlabel('Listing price in $')
#plt.title('No. of bedrooms vs price')

#plt.scatter(df['number_of_reviews'],df['price'])
#plt.ylabel('Listing price in $')
#plt.xlabel('No. of reviews')
#plt.title('No. of reviews vs price')

#df.pivot(columns = 'bedrooms',values = 'price').plot.hist(stacked = True,bins=200)
#plt.xlabel('Listing price in $')

#df.pivot(columns = 'bedrooms',values = 'price').plot.hist(stacked = True,bins=100)
#plt.xlabel('Listing price in $')

# heatmap
#cols = ['number_of_reviews','host_total_listings_count','accommodates','bathrooms','bedrooms','beds','price']

#corrs = np.corrcoef(df[cols].values.T)
#sns.set(font_scale=1)
#hm=sns.heatmap(corrs, cbar = True, annot=True, square = True, fmt = '.2f',
#             yticklabels = cols, xticklabels = cols)


df['host_is_superhost'] = np.where(df['host_is_superhost'] == 't',1,0)
df['instant_bookable'] = np.where(df['instant_bookable'] == 't',1,0)
df['require_guest_profile_picture'] = np.where(df['require_guest_profile_picture'] == 't',1,0)
df['require_guest_phone_verification'] = np.where(df['require_guest_phone_verification'] == 't',1,0)
df['host_response_rate'] = df['host_response_rate'].str.replace("%", "").astype("float")

# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(df['cancellation_policy'])

# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

df = pd.get_dummies(df, columns = ['property_type'])
df = pd.get_dummies(df, columns = ['neighbourhood'])
df = pd.get_dummies(df, columns = ['cancellation_policy'])

clean = {"host_response_time": {"a few days or more":4,"within a day":3,"within a few hours":2,"within an hour":1}}
df.replace(clean,inplace=True)

df['host_since'] = pd.to_datetime(df['host_since'])

df['amenities'] = df['amenities'].str.count(',') + 1
df['host_verifications'] = df['host_verifications'].str.count(',') + 1

df['zipcode'] = df['zipcode'].str[:5]