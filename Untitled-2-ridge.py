import pandas as pd
import numpy as np 
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.utils import shuffle
import math
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error as mabe
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn import tree
from sklearn.feature_selection import RFECV as rfecv
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mabe
from sklearn.metrics import median_absolute_error as mdae
from sklearn.metrics import r2_score as r2
from sklearn.model_selection import cross_val_score
from lassofs import un_imp_features_lasso
from ridgefs import un_imp_features



df1=pd.read_csv('D:\EE 660 Project AIRBNB\Rest.csv')


df1['security_deposit']=df1['security_deposit'].str.replace('$', '')
df1['security_deposit']=df1['security_deposit'].str.replace(',', '')
df1['security_deposit'] = df1['security_deposit'].astype('float64') 

df1['cleaning_fee']=df1['cleaning_fee'].str.replace('$', '')
df1['cleaning_fee']=df1['cleaning_fee'].str.replace(',', '')
df1['cleaning_fee'] = df1['cleaning_fee'].astype('float64') 

df1['extra_people']=df1['extra_people'].str.replace('$', '')
df1['extra_people']=df1['extra_people'].str.replace(',', '')
df1['extra_people'] = df1['extra_people'].astype('float64') 


####DROPPPED!!!!#######
df1=df1.drop('calendar_last_scraped',1)
df1=df1.drop('host_since',1)
df1=df1.drop('host_verifications',1)
df1=df1.drop('zipcode',1)
df1=df1.drop('neighbourhood',1)
df1=df1.drop('host_identity_verified',1)
df1.drop(df1.columns[0],axis=1,inplace=True) #drops the first column, first column is the id number of the shuffled data to select pre-training data

#####DROPPPED######

class DataFrameImputer(TransformerMixin):

    def __init__(self):
        '''
        Impute missing values.

        Columns of dtype object are imputed with the most frequent value 
        in column.

        Columns of other types are imputed with mean of column.
        '''
        
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)


X = pd.DataFrame(df1)
df1 = DataFrameImputer().fit_transform(X)

df1['host_is_superhost'] = np.where(df1['host_is_superhost'] == 't',1,0)
df1['instant_bookable'] = np.where(df1['instant_bookable'] == 't',1,0)
df1['require_guest_profile_picture'] = np.where(df1['require_guest_profile_picture'] == 't',1,0)
df1['require_guest_phone_verification'] = np.where(df1['require_guest_phone_verification'] == 't',1,0)
df1['host_response_rate'] = df1['host_response_rate'].str.replace("%", "").astype("float")

df1 = pd.get_dummies(df1, columns = ['property_type'])
df1 = pd.get_dummies(df1, columns = ['neighbourhood_cleansed'])
df1 = pd.get_dummies(df1, columns = ['cancellation_policy'])

cleanup_data = {"bed_type":     {"Real Bed": 5, "Futon": 4, "Pull-out Sofa":3, "Airbed":1,"Couch":2},
                "room_type": {"Entire home/apt": 3, "Private room": 2, "Shared room": 1}
                }

df1.replace(cleanup_data, inplace=True)


clean = {"host_response_time": {"a few days or more":4,"within a day":3,"within a few hours":2,"within an hour":1}}
df1.replace(clean,inplace=True)


count = df1['amenities'].str.split(",").apply(len)
df1['amenities']=count
#print(df1['amenities'])
df2=df1
for i in range(len(un_imp_features_lasso)):
    if un_imp_features_lasso[i] in df1.columns:
        df2=df2.drop(un_imp_features_lasso[i],axis=1)

for i in range(len(un_imp_features)):
    if un_imp_features[i] in df2.columns:
        df1 = df1.drop(un_imp_features[i], axis=1)

y1=df1['price']
y2=df2['price']

df1=df1.drop('price',1)
df2=df2.drop('price',1)
print('ridge shape',df1.shape)
print('lasso shape', df2.shape)

##Seperating the test set
x_train, x_test, y_train, y_test = train_test_split(df1, y1, test_size=0.20) 
x_train_lass, x_test_lass, y_train_lass, y_test_lass = train_test_split(df2, y2, test_size=0.20) 

'''

#Adaboost
depth=range(1,5)
estimator=[10, 20, 50, 100]
ntrials=5
R_scores_train_lasso=np.zeros((len(depth),len(estimator),ntrials))
R_scores_test_lasso=np.zeros((len(depth),len(estimator),ntrials))
mean_sq_train_lasso=np.zeros((len(depth),len(estimator),ntrials))
mean_sq_test_lasso=np.zeros((len(depth),len(estimator),ntrials))

R_scores_train=np.zeros((len(depth),len(estimator),ntrials))
R_scores_test=np.zeros((len(depth),len(estimator),ntrials))
mean_sq_train=np.zeros((len(depth),len(estimator),ntrials))
mean_sq_test=np.zeros((len(depth),len(estimator),ntrials))

for i in range(len(depth)):
    for k in range(len(estimator)): 
        for j in range(ntrials): 
            x_val_train, x_val_test, y_val_train, y_val_test = train_test_split(x_train, y_train, test_size=0.20) 
            x_val_train_lass, x_val_test_lass, y_val_train_lass, y_val_test_lass = train_test_split(x_train, y_train, test_size=0.20) 
            model = RandomForestRegressor(max_depth=depth[i],n_estimators=estimator[i])
            model.fit(x_val_train,y_val_train)
            y_val_pred = model.predict(x_val_train)
            y_test_pred=model.predict(x_val_test)
            R_scores_train[i][k][j]=r2(y_val_pred,y_val_train)
            R_scores_test[i][k][j]=r2(y_test_pred,y_val_test)
            mean_sq_train[i][k][j]=mse(y_val_train,y_val_pred)
            mean_sq_test[i][k][j]=mse(y_val_test,y_test_pred)
            model.fit(x_val_train_lass,y_val_train_lass)
            y_val_pred_lass = model.predict(x_val_train)
            y_test_pred_lass=model.predict(x_val_test)
            R_scores_train_lasso[i][k][j]=r2(y_val_pred,y_val_train)
            R_scores_test_lasso[i][k][j]=r2(y_test_pred,y_val_test)
            mean_sq_train_lasso[i][k][j]=mse(y_val_train,y_val_pred)
            mean_sq_test_lasso[i][k][j]=mse(y_val_test,y_test_pred)
            



print('Decision tree (Adaboost) : %f' % r2(y_d,y_val_pred))
print('Decision tree (Adaboost) : %f' % mse(y_d,y_val_pred))
print('Decision tree (Adaboost) : %f' % mabe(y_d, y_val_pred))
print('Decision tree (Adaboost) : %f' % mdae(y_d, y_val_pred))

result_train[0,0] = r2(y_d,y_val_pred)
result_train[0,1] = mse(y_d,y_val_pred)
result_train[0,2] = mabe(y_d, y_val_pred)
result_train[0,3] = mdae(y_d, y_val_pred)

#Training Scores
y_train_pred = model.predict(df_d)
result_training[0,0] = r2(y_d,y_train_pred)
result_training[0,1] = mse(y_d,y_train_pred)
result_training[0,2] = mabe(y_d,y_train_pred)
result_training[0,3] = mdae(y_d,y_train_pred)

#Testing set
y_test_pred = model.predict(df_test)
result_test[0,0] = r2(y_test,y_test_pred)
result_test[0,1] = mse(y_test,y_test_pred)
result_test[0,2] = mabe(y_test,y_test_pred)
result_test[0,3] = mdae(y_test,y_test_pred)

#Linear Regression

model = LinearRegression()
model.fit(, y_d)
y_val_pred = model.predict(df_d)
print('Linear Regression : %f' % r2(y_d,y_val_pred))
print('Linear Regression (MSE) : %f' % mse(y_d,y_val_pred))
print('Linear Regression (Mean Absolute Error) : %f' % mabe(y_d, y_val_pred))
print('Linear Regression (Median Absolute Error) : %f' % mdae(y_d, y_val_pred))
result_train[1,0] = r2(y_d,y_val_pred)
result_train[1,1] = mse(y_d,y_val_pred)
result_train[1,2] = mabe(y_d, y_val_pred)
result_train[1,3] = mdae(y_d, y_val_pred)

#Training Scores
y_train_pred = model.predict(df_d)
result_training[1,0] = r2(y_d,y_train_pred)
result_training[1,1] = mse(y_d,y_train_pred)
result_training[1,2] = mabe(y_d,y_train_pred)
result_training[1,3] = mdae(y_d,y_train_pred)

#Testing set
y_test_pred = model.predict(df_d)
result_test[1,0] = r2(y_test,y_test_pred)
result_test[1,1] = mse(y_test,y_test_pred)
result_test[1,2] = mabe(y_test,y_test_pred)
result_test[1,3] = mdae(y_test,y_test_pred)

#Gradient Boosting Regressor

model = GradientBoostingRegressor()
model.fit(df_d, y_d)
y_val_pred = model.predict(df_d)
print('Decision Tree (Gradient boosting) : %f' % r2(y_d,y_val_pred))
print('Decision Tree : Gradient boosting (MSE) : %f' % mse(y_d,y_val_pred))
print('Decision Tree : Gradient boosting (Mean Absolute Error) : %f' % mabe(y_d, y_val_pred))
print('Decision Tree : Gradient boosting (Median Absolute Error) : %f' % mdae(y_d, y_val_pred))
result_train[2,0] = r2(y_d,y_val_pred)
result_train[2,1] = mse(y_d,y_val_pred)
result_train[2,2] = mabe(y_d, y_val_pred)
result_train[2,3] = mdae(y_d, y_val_pred)

#Training Scores
y_train_pred = model.predict(df_d)
result_training[2,0] = r2(y_d,y_train_pred)
result_training[2,1] = mse(y_d,y_train_pred)
result_training[2,2] = mabe(y_d,y_train_pred)
result_training[2,3] = mdae(y_d,y_train_pred)

#Testing set
y_test_pred = model.predict(df_d)
result_test[2,0] = r2(y_test,y_test_pred)
result_test[2,1] = mse(y_test,y_test_pred)
result_test[2,2] = mabe(y_test,y_test_pred)
result_test[2,3] = mdae(y_test,y_test_pred)

#Random Forest Regressor

model = RandomForestRegressor()
model.fit(df_d, y_d)
y_val_pred = model.predict(df_d)
print('Random Forest Regressor : %f' % r2(y_d,y_val_pred))
print('Random Forest Regressor (MSE) : %f' % mse(y_d,y_val_pred))
print('Random Forest Regressor (Mean Absolute Error) : %f' % mabe(y_d, y_val_pred))
print('Random Forest Regressor (Median Absolute Error) : %f' % mdae(y_d, y_val_pred))
result_train[3,0] = r2(y_d,y_val_pred)
result_train[3,1] = mse(y_d,y_val_pred)
result_train[3,2] = mabe(y_d, y_val_pred)
result_train[3,3] = mdae(y_d, y_val_pred)

#Training Scores
y_train_pred = model.predict(df_d)
result_training[3,0] = r2(df_d,y_train_pred)
result_training[3,1] = mse(df_d,y_train_pred)
result_training[3,2] = mabe(df_d,y_train_pred)
result_training[3,3] = mdae(df_d,y_train_pred)

#Testing set
y_test_pred = model.predict(df_test)
result_test[3,0] = r2(y_test,y_test_pred)
result_test[3,1] = mse(y_test,y_test_pred)
result_test[3,2] = mabe(y_test,y_test_pred)
result_test[3,3] = mdae(y_test,y_test_pred)
'''
#XGBoosting Regressor
result_train = np.zeros([5,4])
result_test = np.zeros([5,4])

tree_depth_xgb = np.zeros([13,1])
for val in range(5,18):
    model = XGBRegressor(max_depth=val)
    tree_depth_xgb[val-5,0] = np.mean(cross_val_score(model,x_train,y_train,cv=5))

depthval_xgb = np.argmax(tree_depth_xgb)+5

model = XGBRegressor(max_depth=depthval_xgb)
model.fit(x_train, y_train)
y_val_pred = model.predict(y_test)
print('Decision Tree (XGBoosting) : %f' % r2(y_test,y_val_pred))
print('Decision Tree : XGBoosting (MSE) : %f' % mse(y_test,y_val_pred))
print('Decision Tree : XGBoosting (Mean Absolute Error) : %f' % mabe(y_test, y_val_pred))
print('Decision Tree : XGBoosting (Median Absolute Error) : %f' % mdae(y_test, y_val_pred))
result_test[4,0] = r2(y_test,y_val_pred)
result_test[4,1] = mse(y_test,y_val_pred)
result_test[4,2] = mabe(y_test,y_val_pred)
result_test[4,3] = mdae(y_test,y_val_pred)

#Training Scores
y_train_pred = model.predict(y_train)
result_train[4,0] = r2(y_train,y_train_pred)
result_train[4,1] = mse(y_train,y_train_pred)
result_train[4,2] = mabe(y_train,y_train_pred)
result_train[4,3] = mdae(y_train,y_train_pred)



