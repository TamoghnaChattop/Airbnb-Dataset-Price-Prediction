# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 18:33:58 2018

@author: tchat
"""

import csv
import numpy as np 
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.utils import shuffle
import math
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn import metrics
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn import tree
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn import linear_model
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mabe
from sklearn.metrics import median_absolute_error as mdae
from sklearn.metrics import r2_score as r2
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split as datasplit
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFECV as rfecv
from sklearn.pipeline import Pipeline

# Import Dataset

df = pd.read_csv('D:\EE 660 Project AIRBNB\Exploratory.csv')

# Pre- Processing of Data

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

# Feature Selection

##For lasso or ridge regression we have to standartize the categorical variables since we want our penalty coeffieicents to be fair to all features.

scaler=StandardScaler()

n_alphas = 10
n_trials=10
alphas = np.logspace(-3, 2, n_alphas)
coefs = []
R_scores_train=np.zeros((n_alphas,n_trials))
R_scores_test=np.zeros((n_alphas,n_trials))
mean_sq_train=np.zeros((n_alphas,n_trials))
mean_sq_test=np.zeros((n_alphas,n_trials))

for a in range (len(alphas)):

    for i in range(n_trials):
        
        X_train, X_test, y_train, y_test = datasplit(df, y, test_size=0.1)
        x_train_unorm1 = scaler.fit(X_train)
        ## Normalised data
        x_train_norm1 = scaler.transform(X_train)
        x_test_norm1 = scaler.transform(X_test)
        
        ridge = linear_model.Ridge(alpha=alphas[a], fit_intercept=True)
        ridge.fit(x_train_norm1, y_train)
        y_pred_train=ridge.predict(x_train_norm1)
        y_pred_test=ridge.predict(x_test_norm1)
        coefs.append(ridge.coef_)
        R_scores_train[a][i]=(ridge.score(x_train_norm1,y_train))
        R_scores_test[a][i]=(ridge.score(x_test_norm1,y_test))
        mean_sq_train[a][i]=mse(y_pred_train,y_train)
        mean_sq_test[a][i]=mse(y_pred_test,y_test)
        
mean_accur_test=R_scores_test.mean(1)
mean_accur_train=R_scores_train.mean(1)
std_accur_test_R=np.std(R_scores_test, axis=1)
std_accur_train_R=np.std(R_scores_train, axis=1)
mean_mean_sq_train=mean_sq_train.mean(1)
mean_mean_sq_test=mean_sq_test.mean(1)
std_accur_train_MSE=np.std(mean_sq_train, axis=1)
std_accur_test_MSE=np.std(mean_sq_test, axis=1)

plt.title("Validation Curve with Ridge Regression")
plt.xlabel("alphas")
plt.ylabel("Score")
plt.ylim(0.0, 1.1)
lw = 2
plt.semilogx(alphas, mean_accur_train, label="Training score",
             color="darkorange", lw=lw)
plt.fill_between(alphas, mean_accur_train - std_accur_train_R,
                 mean_accur_train + std_accur_train_R, alpha=0.2,
                 color="darkorange", lw=lw)
plt.semilogx(alphas, mean_accur_test, label="Cross-validation score",
             color="navy", lw=lw)
plt.fill_between(alphas, mean_accur_test - std_accur_test_R,
                 mean_accur_test + std_accur_test_R, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")
plt.show()

index_max_train=np.argmax(mean_accur_train)
index_max_test = np.argmax(mean_accur_test)
index_min_mean_sq_train=np.argmin(mean_mean_sq_train)
index_min_mean_sq_test=np.argmin(mean_mean_sq_test)

print('best R2 alpha:' ,alphas[index_max_test])
print('with mean R2 on test data :' ,mean_accur_test[index_max_test])
print('with mean  R2  on training data :', mean_accur_train[index_max_test])
print('lowest mean squared error alpha: ', alphas[index_min_mean_sq_test])
print('with mean squared error on test data :' ,mean_mean_sq_test[index_min_mean_sq_test])
print('with mean squared error  on training data :', mean_mean_sq_train[index_min_mean_sq_test])
print('best index for R2', index_max_test)
print('best index for Mean Sq Error', index_min_mean_sq_test)
print(np.shape(coefs))
print(alphas)        

######## FINDING THE COEFFICIENTS FOR THE BEST PERFORMING ALPHA ##########

X_train, X_test, y_train, y_test = datasplit(df, y, test_size=0.1)
x_train_unorm1 = scaler.fit(X_train)
        ## Normalised data
x_train_norm1 = scaler.transform(X_train)
x_test_norm1 = scaler.transform(X_test)
ridge2 = linear_model.Ridge(alpha=4.7, fit_intercept=True)
ridge2.fit(x_train_norm1, y_train)

y_pred_train=ridge2.predict(x_train_norm1)
y_pred_test=ridge2.predict(x_test_norm1)

MSE_train=mse(y_pred_train,y_train)
MSE_test=mse(y_pred_test,y_test)

R_score_train=ridge2.score(x_train_norm1,y_train)
R_score_test=ridge2.score(x_test_norm1,y_test)

MAE_train_ridge=mabe(y_pred_train,y_train)
MAE_test_ridge=mabe(y_pred_test,y_test)
MDSE_train_ridge=mdae(y_pred_train,y_train)
MDSE_test_ridge=mdae(y_pred_test,y_test)
weights=ridge2.coef_

print('MSE train ridge', MSE_train)
print('MSE test ridge', MSE_test)

print('R2 train ridge', R_score_train)
print('R2 test ridge', R_score_test)

print('MAE train ridge', MAE_train_ridge)
print('MAE test ridge', MAE_test_ridge)

print('MdAE train ridge', MDSE_train_ridge)
print('MdAE test ridge', MDSE_test_ridge)

lasso2 = linear_model.Lasso(alpha=4.7, fit_intercept=True)
lasso2.fit(x_train_norm1, y_train)
y_pred_train_lasso=lasso2.predict(x_train_norm1)
y_pred_test_lasso=lasso2.predict(x_test_norm1)
MSE_train_lasso=mse(y_pred_train_lasso,y_train)
MSE_test_lasso=mse(y_pred_test_lasso,y_test)
R_score_train_lasso=lasso2.score(x_train_norm1,y_train)
R_score_test_lasso=lasso2.score(x_test_norm1,y_test)
MAE_train_lasso=mabe(y_train,y_pred_train_lasso)
MAE_test_lasso=mabe(y_test,y_pred_test_lasso)
Mdae_train_lasso=mdae(y_pred_train_lasso,y_train)
Mdae_test_lasso=mdae(y_pred_test_lasso,y_test)

weights_lasso=lasso2.coef_
print('MSE train lasso', MSE_train_lasso)
print('MSE test lasso', MSE_test_lasso)
print('R2 train lasso', R_score_train_lasso)
print('R2 test lasso', R_score_test_lasso)
print('MAE train lasso', MAE_train_lasso)
print('MAE test lasso', MAE_test_lasso)
print('MdAE train lasso', Mdae_train_lasso)
print('MdAE test lasso', Mdae_test_lasso)


imp_weigh=np.where(abs(weights)>abs(4),weights,0)
print('important weights wrt to ridge', imp_weigh)

nonzeroind = np.nonzero(imp_weigh)
print(nonzeroind)

features=df.columns
features.tolist()
imp_features=[]
for i in range(0,len(nonzeroind[0])):
    imp_features.append(features[nonzeroind[0][i]])


un_imp_weigh=np.where(abs(weights)<(5),weights,0)
print('unimportant weights ridge', un_imp_weigh)
zeroind = np.where(un_imp_weigh==0)[0]
un_imp_features=[]
for i in range(0,len(zeroind)):
    un_imp_features.append(features[zeroind[i]])

df1 = df.drop(un_imp_features, axis=1)

result_training = np.zeros([5,4])
result_test = np.zeros([5,4])

df1 = preprocessing.normalize(df1)

# Dividing the new dataset into training and test
df_d, df_test, y_d, y_test = datasplit(df1, y, test_size=0.20) 
'''
#Linear Regression

model = LinearRegression()
model.fit(df_d, y_d)
y_test_pred = model.predict(df_test)
print('Linear Regression : %f' % r2(y_test,y_test_pred))
print('Linear Regression (MSE) : %f' % mse(y_test,y_test_pred))
print('Linear Regression (Mean Absolute Error) : %f' % mabe(y_test, y_test_pred))
print('Linear Regression (Median Absolute Error) : %f' % mdae(y_test, y_test_pred))
result_test[0,0] = r2(y_test,y_test_pred)
result_test[0,1] = mse(y_test,y_test_pred)
result_test[0,2] = mabe(y_test,y_test_pred)
result_test[0,3] = mdae(y_test,y_test_pred)
plt.figure()
lims = [np.min([plt.gca().get_xlim(), plt.gca().get_ylim()]), np.max([plt.gca().get_xlim(), plt.gca().get_ylim()])]
plt.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
plt.scatter(y_test, y_test_pred)
plt.ylabel('Target Predicted')
plt.xlabel('True Target')
plt.title('Linear Regression - Test dataset (Ridge Feature Selection)')

#Training Scores
y_train_pred = model.predict(df_d)
result_training[0,0] = r2(y_d,y_train_pred)
result_training[0,1] = mse(y_d,y_train_pred)
result_training[0,2] = mabe(y_d,y_train_pred)
result_training[0,3] = mdae(y_d,y_train_pred)

plt.figure()
lims = [np.min([plt.gca().get_xlim(), plt.gca().get_ylim()]), np.max([plt.gca().get_xlim(), plt.gca().get_ylim()])]
plt.plot(lims, lims, '--k')
plt.scatter(y_d, y_train_pred)
plt.ylabel('Target Predicted')
plt.xlabel('True Target')
plt.title('Linear Regression - Train dataset (Ridge Feature Selection)')
'''
#Adaboost

tree_depth = [3,4,5,6,7,8,9,10,11,12,13,14,15]
MAE_test = np.zeros((len(tree_depth), 5))
MAE_train = np.zeros((len(tree_depth), 5))

for j in range (len(tree_depth)):
    for i in range (5):
        XX_train, XX_test, yy_train, yy_test = datasplit(df_d, y_d, test_size=0.2)

        AB = AdaBoostRegressor(tree.DecisionTreeRegressor(max_depth=tree_depth[j]))
        AB.fit(XX_train, yy_train) 
        yy_pred_train = AB.predict(XX_train)
        yy_pred_test = AB.predict(XX_test)
        MAE_test[j][i]=(mdae(yy_test, yy_pred_test))
        MAE_train[j][i]=(mdae(yy_train, yy_pred_train))

mean_MAE_test = []
print(MAE_test)
for l in range (len(MAE_test)):
    mean_MAE_test.append(np.mean(MAE_test[l:,]))
    
depthval = np.argmin(mean_MAE_test)     

n_est = [10,100,200,300,500]
MAE_test = np.zeros((len(n_est), 5))
MAE_train = np.zeros((len(n_est), 5))

for j in range (len(n_est)):
    for i in range (5):
        XX_train, XX_test, yy_train, yy_test = datasplit(df_d, y_d, test_size=0.2)

        AB = AdaBoostRegressor(n_estimators=n_est[j])
        AB.fit(XX_train, yy_train) 
        yy_pred_train = AB.predict(XX_train)
        yy_pred_test = AB.predict(XX_test)
        MAE_test[j][i]=(mdae(yy_test, yy_pred_test))
        MAE_train[j][i]=(mdae(yy_train, yy_pred_train))

mean_MAE_test = []
print(MAE_test)

for l in range (len(MAE_test)):
    mean_MAE_test.append(np.mean(MAE_test[l:,]))

ne=np.argmin(mean_MAE_test)
print(ne)

model = AdaBoostRegressor(tree.DecisionTreeRegressor(max_depth=tree_depth[depthval]), n_estimators = n_est[ne])
model.fit(df_d, y_d)
y_test_pred = model.predict(df_test)
print('Decision tree (Adaboost) : %f' % r2(y_test,y_test_pred))
print('Decision tree (Adaboost) : %f' % mse(y_test,y_test_pred))
print('Decision tree (Adaboost) : %f' % mabe(y_test, y_test_pred))
print('Decision tree (Adaboost) : %f' % mdae(y_test, y_test_pred))
result_test[1,0] = r2(y_test,y_test_pred)
result_test[1,1] = mse(y_test,y_test_pred)
result_test[1,2] = mabe(y_test,y_test_pred)
result_test[1,3] = mdae(y_test,y_test_pred)
plt.figure()
plt.plot(y_test_pred, 'r', label = 'pred') 
plt.plot(y_test.values, 'b', label = 'true')
plt.ylabel('Number of data points')
plt.xlabel('Price')
plt.title('AdaBoost Regression - Test dataset (Ridge Feature Selection)')

#Training Scores
y_train_pred = model.predict(df_d)
result_training[1,0] = r2(y_d,y_train_pred)
result_training[1,1] = mse(y_d,y_train_pred)
result_training[1,2] = mabe(y_d,y_train_pred)
result_training[1,3] = mdae(y_d,y_train_pred)
plt.figure()
plt.plot(y_train_pred, 'r', label = 'pred') 
plt.plot(y_d.values, 'b', label = 'true')
plt.ylabel('Number of data points')
plt.xlabel('Price')
plt.title('AdaBoost Regression - Train dataset (Ridge Feature Selection)')


'''
#Gradient Boosting Regressor

model = GradientBoostingRegressor()
model.fit(df_d, y_d)
y_test_pred = model.predict(df_test)
print('Decision Tree (Gradient boosting) : %f' % r2(y_test,y_test_pred))
print('Decision Tree : Gradient boosting (MSE) : %f' % mse(y_test,y_test_pred))
print('Decision Tree : Gradient boosting (Mean Absolute Error) : %f' % mabe(y_test, y_test_pred))
print('Decision Tree : Gradient boosting (Median Absolute Error) : %f' % mdae(y_test, y_test_pred))
result_test[2,0] = r2(y_test,y_test_pred)
result_test[2,1] = mse(y_test,y_test_pred)
result_test[2,2] = mabe(y_test,y_test_pred)
result_test[2,3] = mdae(y_test,y_test_pred)

#Training Scores
y_train_pred = model.predict(df_d)
result_training[2,0] = r2(y_d,y_train_pred)
result_training[2,1] = mse(y_d,y_train_pred)
result_training[2,2] = mabe(y_d,y_train_pred)
result_training[2,3] = mdae(y_d,y_train_pred)

#Random Forest Regressor

#n_est = [10,100,200,300,500]
#MAE_test = np.zeros((len(n_est), 5))
#MAE_train = np.zeros((len(n_est), 5))

#for j in range (len(n_est)):
#    for i in range (5):
#        XX_train, XX_test, yy_train, yy_test = datasplit(df1, y, test_size=0.2)

#        RF = RandomForestRegressor(n_estimators=n_est[j], criterion='mse')
#        RF.fit(XX_train, yy_train) 
#        yy_pred_train = RF.predict(XX_train)
#        yy_pred_test = RF.predict(XX_test)
#        MAE_test[j][i]=(mdae(yy_test, yy_pred_test))
#        MAE_train[j][i]=(mdae(yy_train, yy_pred_train))

#mean_MAE_test = []
#for l in range (len(MAE_test)):
#    mean_MAE_test.append(np.mean(MAE_test[l:,]))

#model = RandomForestRegressor(n_estimators=np.argmin(mean_MAE_test), criterion='mse')
#model.fit(df_d, y_d)
#y_test_pred = model.predict(df_test)
#print('Random Forest Regressor : %f' % r2(y_test,y_test_pred))
#print('Random Forest Regressor (MSE) : %f' % mse(y_test,y_test_pred))
#print('Random Forest Regressor (Mean Absolute Error) : %f' % mabe(y_test, y_test_pred))
#print('Random Forest Regressor (Median Absolute Error) : %f' % mdae(y_test, y_test_pred))
#result_test[3,0] = r2(y_test,y_test_pred)
#result_test[3,1] = mse(y_test,y_test_pred)
#result_test[3,2] = mabe(y_test,y_test_pred)
#result_test[3,3] = mdae(y_test,y_test_pred)

#Training Scores
#y_train_pred = model.predict(df_d)
#result_training[3,0] = r2(y_d,y_train_pred)
#result_training[3,1] = mse(y_d,y_train_pred)
#result_training[3,2] = mabe(y_d,y_train_pred)
#result_training[3,3] = mdae(y_d,y_train_pred)
'''

#XGBoosting Regressor

tree_depth_xgb = np.zeros([13,1])
for val in range(5,18):
    model = XGBRegressor(max_depth=val)
    tree_depth_xgb[val-5,0] = np.mean(cross_val_score(model,df_d,y_d,cv=5))

depthval_xgb = np.argmax(tree_depth_xgb)+5

model = XGBRegressor(max_depth=depthval_xgb)
model.fit(df_d, y_d)
y_test_pred = model.predict(df_test)
print('Decision Tree (XGBoosting) : %f' % r2(y_test,y_test_pred))
print('Decision Tree : XGBoosting (MSE) : %f' % mse(y_test,y_test_pred))
print('Decision Tree : XGBoosting (Mean Absolute Error) : %f' % mabe(y_test, y_test_pred))
print('Decision Tree : XGBoosting (Median Absolute Error) : %f' % mdae(y_test, y_test_pred))
result_test[4,0] = r2(y_test,y_test_pred)
result_test[4,1] = mse(y_test,y_test_pred)
result_test[4,2] = mabe(y_test,y_test_pred)
result_test[4,3] = mdae(y_test,y_test_pred)
plt.figure()
plt.plot(y_test_pred, 'r', label = 'pred') 
plt.plot(y_test.values, 'b', label = 'true')
plt.ylabel('Number of data points')
plt.xlabel('Price')
plt.title('XGBoost Regression - Test dataset (Ridge Feature Selection)')

#Training Scores
y_train_pred = model.predict(df_d)
result_training[4,0] = r2(y_d,y_train_pred)
result_training[4,1] = mse(y_d,y_train_pred)
result_training[4,2] = mabe(y_d,y_train_pred)
result_training[4,3] = mdae(y_d,y_train_pred)
print('MSE train xgboost lasso', result_training[4,1])
print('R2 train xgboost lasso', result_training[4,0])
print('MAE train xgboost lasso', result_training[4,2])
print('MdAE train xgboost lasso', result_training[4,3])
plt.figure()
plt.plot(y_train_pred, 'r', label = 'pred') 
plt.plot(y_d.values, 'b', label = 'true')
plt.ylabel('Number of data points')
plt.xlabel('Price')
plt.title('XGBoost Regression - Train dataset (Ridge Feature Selection)')

