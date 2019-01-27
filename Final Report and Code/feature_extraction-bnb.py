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
from sklearn import tree
from sklearn.feature_selection import RFECV as rfecv
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mabe
from sklearn.metrics import median_absolute_error as mdae
from sklearn.metrics import r2_score as r2
from sklearn.model_selection import cross_val_score

df=pd.read_csv('/Users/Penguin/Desktop/Exploratory.csv')
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
print(df['amenities'])

#df.to_csv(path_or_buf='/Users/Penguin/Desktop/Exp-Processed.csv')

###Model Selection and exploration for Pretraining data##

y=df['price']
df=df.drop('price',1)


##For lasso or ridge regression we have to standartize the categorical variables since we want our penalty coeffieicents to be fair to all features.

scaler=StandardScaler()
'''
n_alphas = 10
n_trials=40
alphas = np.logspace(-2, 3, n_alphas)
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
        
        X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2)
        x_train_unorm1 = scaler.fit(X_train)
        ## Standartized data
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
plt.semilogx(alphas, mean_accur_train_lasso, label="Training score",
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


plt.title("Validation Curve with Ridge Regression")
plt.xlabel("alphas")
plt.ylabel("MSE Score")
lw = 2
plt.semilogx(alphas, mean_mean_sq_train, label="Training score",
             color="darkorange", lw=lw)
plt.fill_between(alphas, mean_mean_sq_train - std_accur_train_MSE,
                 mean_mean_sq_train + std_accur_train_MSE, alpha=0.2,
                 color="darkorange", lw=lw)
plt.semilogx(alphas, mean_mean_sq_test, label="Cross-validation score",
             color="navy", lw=lw)
plt.fill_between(alphas, mean_mean_sq_test - std_accur_test_MSE,
                 mean_mean_sq_test + std_accur_test_MSE, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")
plt.show()



plt.title("Validation Curve with Lasso Regression")
plt.xlabel("alphas")
plt.ylabel("MSE Score")
lw = 2
plt.semilogx(alphas, mean_mean_sq_train_lasso, label="Training score",
             color="darkorange", lw=lw)
plt.fill_between(alphas, mean_mean_sq_train_lasso - std_accur_train_MSE_lasso,
                 mean_mean_sq_train_lasso + std_accur_train_MSE_lasso, alpha=0.2,
                 color="darkorange", lw=lw)
plt.semilogx(alphas, mean_mean_sq_test_lasso, label="Cross-validation score",
             color="navy", lw=lw)
plt.fill_between(alphas, mean_mean_sq_test_lasso - std_accur_test_MSE_lasso,
                 mean_mean_sq_test_lasso + std_accur_test_MSE_lasso, alpha=0.2,
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

'''

######## FINDING THE COEFFICIENTS FOR THE BEST PERFORMING ALPHA ##########

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2)
x_train_unorm1 = scaler.fit(X_train)
        ## standartized data
x_train_norm1 = scaler.transform(X_train)
x_test_norm1 = scaler.transform(X_test)
ridge2 = linear_model.Ridge(alpha=4.7, fit_intercept=True)
ridge2.fit(x_train_norm1, y_train)
lasso2 = linear_model.Lasso(alpha=4.7, fit_intercept=True)
lasso2.fit(x_train_norm1, y_train)
y_pred_train=ridge2.predict(x_train_norm1)
y_pred_test=ridge2.predict(x_test_norm1)
y_pred_train_lasso=lasso2.predict(x_train_norm1)
y_pred_test_lasso=lasso2.predict(x_test_norm1)
MSE_train=mean_squared_error(y_pred_train,y_train)
MSE_test=mean_squared_error(y_pred_test,y_test)
MSE_train_lasso=mean_squared_error(y_pred_train_lasso,y_train)
MSE_test_lasso=mean_squared_error(y_pred_test_lasso,y_test)
R_score_train=ridge2.score(x_train_norm1,y_train)
R_score_test=ridge2.score(x_test_norm1,y_test)
R_score_train_lasso=lasso2.score(x_train_norm1,y_train)
R_score_test_lasso=lasso2.score(x_test_norm1,y_test)
MAE_train_ridge=mabe(y_pred_train,y_train)
MAE_test_ridge=mabe(y_pred_test,y_test)
MAE_train_lasso=mabe(y_pred_train_lasso,y_pred_train)
MAE_test_lasso=mabe(y_pred_test_lasso,y_pred_test)

weights=ridge2.coef_
weights_lasso=lasso2.coef_
print('MSE train ridge', MSE_train)
print('MSE test ridge', MSE_test)
print('MSE train lasso', MSE_train_lasso)
print('MSE test lasso', MSE_test_lasso)
print('R2 train ridge', R_score_train)
print('R2 test ridge', R_score_test)
print('R2 train lasso', R_score_train_lasso)
print('R2 test lasso', R_score_test_lasso)
print('MAE train ridge', MAE_train_ridge)
print('MAE test ridge', MAE_test_ridge)
print('MAE train lasso', MAE_train_lasso)
print('MAE test lasso', MAE_test_lasso)
#print('weights :', weights)

'''
plt.plot(weights)
plt.ylabel('Weight value')
plt.xlabel('weights')
plt.show()
'''
imp_weigh=np.where(abs(weights)>abs(4),weights,0)
print('important weights wrt to ridge', imp_weigh)

nonzeroind = np.nonzero(imp_weigh)
print(nonzeroind)

features=df.columns
features.tolist()
imp_features=[]
for i in range(0,len(nonzeroind[0])):
    imp_features.append(features[nonzeroind[0][i]])
print('important features wrt to ridge: ')
print(imp_features)
print(len(imp_features))



imp_weigh_lasso=np.where(abs(weights_lasso)>abs(4),weights_lasso,0)
print('important weights wrt to lasso', imp_weigh_lasso)

nonzeroind = np.nonzero(imp_weigh_lasso)
imp_features_lasso=[]
for i in range(0,len(nonzeroind[0])):
    imp_features_lasso.append(features[nonzeroind[0][i]])
print('important features wrt to lasso: ')
print(imp_features_lasso)
print(len(imp_features_lasso))

un_imp_weigh=np.where(abs(weights)<1,0,10)
print('unimportant weights ridge', un_imp_weigh)
zeroind = np.where(un_imp_weigh==0)[0]
un_imp_features=[]
for i in range(0,len(zeroind)):
    un_imp_features.append(features[zeroind[i]])
print('unimportant features wrt ridge')
print(un_imp_features)
print(len(un_imp_features))

un_imp_weigh_lasso=np.where(abs(weights_lasso)<(1),0,10)
print('unimportant weights lasso', un_imp_weigh_lasso)
zeroind = np.where(un_imp_weigh_lasso==0)[0]
un_imp_features_lasso=[]
for i in range(0,len(zeroind)):
    un_imp_features_lasso.append(features[zeroind[i]])

print('unimportant features wrt lasso')
print(un_imp_features_lasso)
print(len(un_imp_features_lasso))

'''
df = df.drop(un_imp_features_lasso, axis=1)


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


n_est = [10,100,200,300,500]
MAE_test = np.zeros((len(n_est), 5))
MAE_train = np.zeros((len(n_est), 5))

for j in range (len(n_est)):
    for i in range (5):
        XX_train, XX_test, yy_train, yy_test = train_test_split(df, y, test_size=0.2)
        RF = RandomForestRegressor(n_estimators=n_est[j], criterion='mse')
        RF.fit(XX_train, yy_train) 
        yy_pred_train = RF.predict(XX_train)
        yy_pred_test = RF.predict(XX_test)
        MAE_test[j][i]=(mdae(yy_test, yy_pred_test))
        MAE_train[j][i]=(mdae(yy_train, yy_pred_train))

mean_MAE_test = []
for l in range (len(MAE_test)):
    mean_MAE_test.append(np.mean(MAE_test[l:,]))

RF2 = RandomForestRegressor(n_estimators=np.argmin(mean_MAE_test), criterion='mse')
RF2.fit(XX_train, yy_train) 
yy_pred_train2 = RF.predict(XX_train)
yy_pred_test2 = RF.predict(XX_test)
MAE_final = mdae(yy_test, yy_pred_test2)
print(MAE_final)




df1=pd.read_csv('/Users/Penguin/Desktop/OS/USC COURSES/EE 660/Project/Rest.csv')

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
       
        Impute missing values.

        Columns of dtype object are imputed with the most frequent value 
        in column.

        Columns of other types are imputed with mean of column.
        
        
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
print(df['amenities'])


df1 = df1.drop(un_imp_features, axis=1)
df1.to_csv('/Users/Penguin/Desktop/Rest-Processed.csv')
y=df1['price']
df1=df1.drop('price',1)
print(df1.shape())
result_train = np.zeros([5,4])
result_training = np.zeros([5,4])
result_test = np.zeros([5,4])

#Adaboost

df_d, df_test, y_d, y_test = train_test_split(df1, y, test_size=0.20) 

tree_depth = np.zeros([18,1])
for val in range(2,20):
    model = tree.DecisionTreeRegressor(max_depth=val)
    tree_depth[val-2,0] = np.mean(cross_val_score(model,df_d,y_d,cv=5))

depthval = np.argmax(tree_depth)+2

model = AdaBoostRegressor(tree.DecisionTreeRegressor(max_depth=depthval))
model.fit(df_d, y_d)

y_val_pred = model.predict(df_d)
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
model.fit(df_d, y_d)
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
result_training[1,0] = r2(y_train,y_train_pred)
result_training[1,1] = mse(y_train,y_train_pred)
result_training[1,2] = mabe(y_train,y_train_pred)
result_training[1,3] = mdae(y_train,y_train_pred)

#Testing set
y_test_pred = model.predict(df_d)
result_test[1,0] = r2(y_test,y_test_pred)
result_test[1,1] = mse(y_test,y_test_pred)
result_test[1,2] = mabe(y_test,y_test_pred)
result_test[1,3] = mdae(y_test,y_test_pred)

#Gradient Boosting Regressor

model = GradientBoostingRegressor()
model.fit(df_d, y_train)
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
result_training[2,0] = r2(y_train,y_train_pred)
result_training[2,1] = mse(y_train,y_train_pred)
result_training[2,2] = mabe(y_train,y_train_pred)
result_training[2,3] = mdae(y_train,y_train_pred)

#Testing set
y_test_pred = model.predict(df_d)
result_test[2,0] = r2(y_test,y_test_pred)
result_test[2,1] = mse(y_test,y_test_pred)
result_test[2,2] = mabe(y_test,y_test_pred)
result_test[2,3] = mdae(y_test,y_test_pred)

#Random Forest Regressor

model = RandomForestRegressor()
model.fit(df_d, y_train)
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

#XGBoosting Regressor

tree_depth_xgb = np.zeros([13,1])
for val in range(5,18):
    model = XGBRegressor(max_depth=val)
    tree_depth_xgb[val-5,0] = np.mean(cross_val_score(model,df_d,y_d,cv=5))

depthval_xgb = np.argmax(tree_depth_xgb)+5

model = XGBRegressor(max_depth=depthval_xgb)
model.fit(df_train, y_train)
y_val_pred = model.predict(df_val)
print('Decision Tree (XGBoosting) : %f' % r2(y_val,y_val_pred))
print('Decision Tree : XGBoosting (MSE) : %f' % mse(y_val,y_val_pred))
print('Decision Tree : XGBoosting (Mean Absolute Error) : %f' % mabe(y_val, y_val_pred))
print('Decision Tree : XGBoosting (Median Absolute Error) : %f' % mdae(y_val, y_val_pred))
result_train[4,0] = r2(y_val,y_val_pred)
result_train[4,1] = mse(y_val,y_val_pred)
result_train[4,2] = mabe(y_val, y_val_pred)
result_train[4,3] = mdae(y_val, y_val_pred)

#Training Scores
y_train_pred = model.predict(df_d)
result_training[4,0] = r2(y_train,y_train_pred)
result_training[4,1] = mse(y_train,y_train_pred)
result_training[4,2] = mabe(y_train,y_train_pred)
result_training[4,3] = mdae(y_train,y_train_pred)

#Testing set
y_test_pred = model.predict(df_test)
result_test[4,0] = r2(y_test,y_test_pred)
result_test[4,1] = mse(y_test,y_test_pred)
result_test[4,2] = mabe(y_test,y_test_pred)
result_test[4,3] = mdae(y_test,y_test_pred)

'''