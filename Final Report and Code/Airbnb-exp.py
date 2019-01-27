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
from sklearn import metrics
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import os
cwd = os.getcwd()
print(cwd)
df = pd.read_csv('‎⁨⁨/Users⁩/Penguin⁩/Desktop⁩/OS⁩/USC COURSES⁩/⁨EE 660⁩/Project⁩/Exploratory.csv')


df['security_deposit']=df['security_deposit'].str.replace('$', '')
df['security_deposit']=df['security_deposit'].str.replace(',', '')
df['security_deposit'] = df['security_deposit'].astype('float64') 

df['cleaning_fee']=df['cleaning_fee'].str.replace('$', '')
df['cleaning_fee']=df['cleaning_fee'].str.replace(',', '')
df['cleaning_fee'] = df['cleaning_fee'].astype('float64') 

df['extra_people']=df['extra_people'].str.replace('$', '')
df['extra_people']=df['extra_people'].str.replace(',', '')
df['extra_people'] = df['extra_people'].astype('float64') 
####DROPPPED!!!!#######
df=df.drop('calendar_last_scraped',1)
df=df.drop('host_since',1)
df=df.drop('host_verifications',1)
df=df.drop('zipcode',1)
df=df.drop('neighbourhood',1)
df=df.drop('host_identity_verified',1)
df.drop(df.columns[0],axis=1,inplace=True) #drops the first column, first column is the id number of the shuffled data to select pre-training data

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

X = pd.DataFrame(df)
df = DataFrameImputer().fit_transform(X)


#x=(df.groupby('neighbourhood_cleansed')['price'].sum()) # this code is used for fj dj g the average price per neighborhood
#y=(df['neighbourhood_cleansed'].value_counts(sort=False))
#print(x)
#print(y)

#x.to_csv('/Users/Penguin/Desktop/x.csv')
#y.to_csv('/Users/Penguin/Desktop/y.csv')

df['host_is_superhost'] = np.where((df['host_is_superhost']) == 't',1,0)
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
print(df['amenities'])

df.to_csv(path_or_buf='/Users/Penguin/Desktop/Exp-Processed.csv')

###Model Selection and exploration for Pretraining data##

y=df['price']
df=df.drop('price',1)


##For lasso or ridge regression we have to standartize the categorical variables since we want our penalty coeffieicents to be fair to all features.

scaler=StandardScaler()

n_alphas = 10
n_trials=10
alphas = np.logspace(-4, 2, n_alphas)
coefs = []
coefs_lasso=[]
R_scores_train_lasso=np.zeros((n_alphas,n_trials))
R_scores_test_lasso=np.zeros((n_alphas,n_trials))
mean_sq_train_lasso=np.zeros((n_alphas,n_trials))
mean_sq_test_lasso=np.zeros((n_alphas,n_trials))

R_scores_train=np.zeros((n_alphas,n_trials))
R_scores_test=np.zeros((n_alphas,n_trials))
mean_sq_train=np.zeros((n_alphas,n_trials))
mean_sq_test=np.zeros((n_alphas,n_trials))


for a in range (len(alphas)):

    for i in range(n_trials):
        
        X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.1)
        x_train_unorm1 = scaler.fit(X_train)
        ## Normalised data
        x_train_norm1 = scaler.transform(X_train)
        x_test_norm1 = scaler.transform(X_test)
        lasso = linear_model.Lasso(alpha=alphas[a], fit_intercept=True)
        lasso.fit(x_train_norm1, y_train)
        y_pred_train_lasso=lasso.predict(x_train_norm1)
        y_pred_test_lasso=lasso.predict(x_test_norm1)
        coefs_lasso.append(lasso.coef_)
        R_scores_train_lasso[a][i]=(lasso.score(x_train_norm1,y_train))
        R_scores_test_lasso[a][i]=(lasso.score(x_test_norm1,y_test))
        mean_sq_train_lasso[a][i]=mean_squared_error(y_pred_train_lasso,y_train)
        mean_sq_test_lasso[a][i]=mean_squared_error(y_pred_test_lasso,y_test)

        ridge = linear_model.Ridge(alpha=alphas[a], fit_intercept=True)
        ridge.fit(x_train_norm1, y_train)
        y_pred_train=ridge.predict(x_train_norm1)
        y_pred_test=ridge.predict(x_test_norm1)
        coefs.append(ridge.coef_)
        R_scores_train[a][i]=(ridge.score(x_train_norm1,y_train))
        R_scores_test[a][i]=(ridge.score(x_test_norm1,y_test))
        mean_sq_train[a][i]=mean_squared_error(y_pred_train,y_train)
        mean_sq_test[a][i]=mean_squared_error(y_pred_test,y_test)


mean_accur_test_lasso=R_scores_test_lasso.mean(1)
mean_accur_train_lasso=R_scores_train_lasso.mean(1)
std_accur_test_R_lasso=np.std(R_scores_test_lasso, axis=1)
std_accur_train_R_lasso=np.std(R_scores_train_lasso, axis=1)
mean_mean_sq_train_lasso=mean_sq_train_lasso.mean(1)
mean_mean_sq_test_lasso=mean_sq_test_lasso.mean(1)
std_accur_train_MSE_lasso=np.std(mean_sq_train_lasso, axis=1)
std_accur_test_MSE_lasso=np.std(mean_sq_test_lasso, axis=1)




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


plt.title("Validation Curve with Lasso Regression")
plt.xlabel("alphas")
plt.ylabel("Score")
plt.ylim(0.0, 1.1)
lw = 2
plt.semilogx(alphas, mean_accur_train, label="Training score",
             color="darkorange", lw=lw)
plt.fill_between(alphas, mean_accur_train_lasso - std_accur_train_R_lasso,
                 mean_accur_train_lasso + std_accur_train_R_lasso, alpha=0.2,
                 color="darkorange", lw=lw)
plt.semilogx(alphas, mean_accur_test_lasso, label="Cross-validation score",
             color="navy", lw=lw)
plt.fill_between(alphas, mean_accur_test_lasso - std_accur_test_R_lasso,
                 mean_accur_test_lasso + std_accur_test_R_lasso, alpha=0.2,
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

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.1)
x_train_unorm1 = scaler.fit(X_train)
        ## Normalised data
x_train_norm1 = scaler.transform(X_train)
x_test_norm1 = scaler.transform(X_test)
ridge2 = linear_model.Ridge(alpha=2.15443469e-03, fit_intercept=True)
ridge2.fit(x_train_norm1, y_train)
y_pred_train=ridge2.predict(x_train_norm1)
y_pred_test=ridge2.predict(x_test_norm1)
MSE_train=mean_squared_error(y_pred_train,y_train)
MSE_test=mean_squared_error(y_pred_test,y_test)

weights=ridge2.coef_
print('MSE train', MSE_train)
print('MSE test', MSE_test)
print('weights :', weights)

'''
plt.plot(weights)
plt.ylabel('Weight value')
plt.xlabel('weights')
plt.show()
'''
imp_weigh=np.where(weights>abs(4),weights,0)
print('important weights', imp_weigh)

nonzeroind = np.nonzero(imp_weigh)
print(nonzeroind)

features=df.columns
features.tolist()
imp_features=[]
for i in range(0,len(nonzeroind[0])):
    imp_features.append(features[nonzeroind[0][i]])
    
print(imp_features)
print(len(imp_features))


un_imp_weigh=np.where(weights<abs(1),weights,0)
print('unimportant weights', un_imp_weigh)
nonzeroind = np.nonzero(un_imp_weigh)
un_imp_features=[]
for i in range(0,len(nonzeroind[0])):
    un_imp_features.append(features[nonzeroind[0][i]])

print(un_imp_features)
print(len(un_imp_features))



X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.1)

poly = PolynomialFeatures(degree=2)
x_train_expanded=poly.fit_transform(X_train)
x_test_expanded=poly.fit_transform(X_test)
reg = LinearRegression().fit(X_train, y_train)
R_scores_train=reg.score(X_train,y_train)

R_score_test=reg.score(X_test,y_test)
y_pred_train_l=reg.predict(X_train)
y_pred_test_l=reg.predict(X_test)
MSE_train_l=mean_squared_error(y_pred_train_l,y_train)
MSE_test_l=mean_squared_error(y_pred_test_l,y_test)
print(R_score_test)
print(R_scores_train)
print(MSE_train_l)
print(MSE_test_l)
