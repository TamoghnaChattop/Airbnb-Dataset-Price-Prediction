import csv
import numpy as np 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from sklearn.utils import shuffle
import math
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.base import TransformerMixin

df2 = pd.read_csv('/Users/Penguin/Desktop/OS/USC COURSES/EE 660/Project/listings.csv', parse_dates=True)
df2['price']=df2['price'].str.replace('$', '')
df2['price']=df2['price'].str.replace(',', '')
df2['price'] = df2['price'].astype('float64') 

#print(df2['price'])
#print(df2.shape)
#print(df2.columns)
df2=df2.drop('listing_url',1)
df2=df2.drop('scrape_id',1)
df2=df2.drop('experiences_offered',1)
df2=df2.drop('last_scraped',1)
df2=df2.drop('thumbnail_url',1)
df2=df2.drop('medium_url',1)
df2=df2.drop('picture_url',1)
df2=df2.drop('xl_picture_url',1)
df2=df2.drop('host_picture_url',1)
df2=df2.drop('host_url',1)
df2=df2.drop('host_name',1)
df2=df2.drop('host_thumbnail_url',1)
df2=df2.drop('host_has_profile_pic',1)
df2=df2.drop('neighbourhood_group_cleansed',1)
df2=df2.drop('market',1)
df2=df2.drop('is_location_exact',1)
df2=df2.drop('weekly_price',1)
df2=df2.drop('monthly_price',1)
df2=df2.drop('is_business_travel_ready',1)
df2=df2.drop('jurisdiction_names',1)
df2=df2.drop('license',1)
df2=df2.drop('requires_license',1)
df2=df2.drop('name',1)
df2=df2.drop('summary',1)
df2=df2.drop('space',1)
df2=df2.drop('description',1)
df2=df2.drop('host_id',1)
df2=df2.drop('host_neighbourhood',1)
df2=df2.drop('host_location',1)
df2=df2.drop('host_listings_count',1)
df2=df2.drop('street',1)
df2=df2.drop('city',1)
df2=df2.drop('smart_location',1)
df2=df2.drop('state',1)
df2=df2.drop('country',1)
df2=df2.drop('country_code',1)
df2=df2.drop('calendar_updated',1)
df2=df2.drop('has_availability',1)
df2=df2.drop('availability_30',1)
df2=df2.drop('availability_60',1)
df2=df2.drop('availability_90',1)
df2=df2.drop('availability_365',1)
df2=df2.drop('first_review',1)
df2=df2.drop('last_review',1)

####DROPPPED!!!!#######
df2=df2.drop('host_since',1)
df2=df2.drop('calendar_last_scraped',1)
#####DROPPPED######


#print(df2.shape)
#print(df2.isnull().sum(axis=0))
nully1=df2.isnull().sum(axis=0) #to decide which features have the most missing points and deleting the ones with having more then 30% of missing points
#print(nully1)
#print(np.where(nully1>13000))#to decide which features have missing data more than 13000.(30% of the data)

df2=df2.drop('neighborhood_overview',1)
df2=df2.drop('notes',1)
df2=df2.drop('transit',1)
df2=df2.drop('access',1)
df2=df2.drop('interaction',1)
df2=df2.drop('house_rules',1)

df2=df2.drop('host_about',1)
df2=df2.drop('host_acceptance_rate',1)
df2=df2.drop('square_feet',1)
#print(df2.shape)

nully2=df2.isnull().sum(axis=1) #to determine nulls for rows
n=np.where(nully2>10)
#print(n)

df2=df2.drop(df2.index[n])
       

#print(df2.shape)



df2=df2.replace(to_replace='deleted', value=np.NaN) #some of the missing values are labeled as 'deleted' in the document

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

X = pd.DataFrame(df2)
xt = DataFrameImputer().fit_transform(X)
xt.to_csv(path_or_buf='/Users/Penguin/Desktop/out.csv')

#df3=df2.as_matrix()


#id=np.random.permutation(len(df3))

sf=df2.iloc[np.random.permutation(len(df2))]
exp = sf.iloc[0:5000,:]
rest = sf.iloc[5001:,:]

#exp = sf[0:5000,:] #first 5000 points of the shuffled data is 
#rest=sf[5001:40384,:] #rest is for training cross val. and testing

#exp = pd.DataFrame(exp)
#rest = pd.DataFrame(rest)

exp.to_csv(path_or_buf='/Users/Penguin/Desktop/Exploratory.csv')
rest.to_csv(path_or_buf='/Users/Penguin/Desktop/Rest.csv')

