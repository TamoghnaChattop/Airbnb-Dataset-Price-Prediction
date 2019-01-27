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
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn.feature_selection import RFECV as rfecv
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mabe
from sklearn.metrics import median_absolute_error as mdae
from sklearn.metrics import r2_score as r2
from sklearn.model_selection import cross_val_score
from Untitled_bnb import un_imp_features, un_imp_features_lasso
from numpy import unravel_index
from sklearn.model_selection import KFold 




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
    if un_imp_features_lasso[i] in df2.columns:
        df2=df2.drop(un_imp_features_lasso[i],axis=1)

for i in range(len(un_imp_features)):
    if un_imp_features[i] in df1.columns:
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




#Random Forest Regressor
depth=range(3,6)
estimator=[10, 50, 100, 200]
ntrials=8
'''

R_scores_train_lasso=np.zeros((len(depth),len(estimator),ntrials))
R_scores_test_lasso=np.zeros((len(depth),len(estimator),ntrials))
mean_sq_train_lasso=np.zeros((len(depth),len(estimator),ntrials))
mean_sq_test_lasso=np.zeros((len(depth),len(estimator),ntrials))
mean_abs_test_lasso=np.zeros((len(depth),len(estimator),ntrials))
mean_abs_train_lasso=np.zeros((len(depth),len(estimator),ntrials))
median_abs_train_lasso=np.zeros((len(depth),len(estimator),ntrials))
median_abs_tests_lasso=np.zeros((len(depth),len(estimator),ntrials))


R_scores_train=np.zeros((len(depth),len(estimator),ntrials))
R_scores_test=np.zeros((len(depth),len(estimator),ntrials))
mean_sq_train=np.zeros((len(depth),len(estimator),ntrials))
mean_sq_test=np.zeros((len(depth),len(estimator),ntrials))
mean_abs_train=np.zeros((len(depth),len(estimator),ntrials))
mean_abs_test=np.zeros((len(depth),len(estimator),ntrials))
median_abs_train=np.zeros((len(depth),len(estimator),ntrials))
median_abs_test=np.zeros((len(depth),len(estimator),ntrials))



for i in range(len(depth)):
    for k in range(len(estimator)): 
        for j in range(ntrials): 
            x_val_train, x_val_test, y_val_train, y_val_test = train_test_split(x_train, y_train, test_size=0.20) 
            x_val_train_lass, x_val_test_lass, y_val_train_lass, y_val_test_lass = train_test_split(x_train_lass, y_train_lass, test_size=0.20) 
            model = RandomForestRegressor(max_depth=depth[i],n_estimators=estimator[k])
            model.fit(x_val_train,y_val_train)

            y_val_pred = model.predict(x_val_train)
            y_test_pred=model.predict(x_val_test)

            R_scores_train[i][k][j]=r2(y_val_pred,y_val_train)
            R_scores_test[i][k][j]=r2(y_test_pred,y_val_test)
            mean_sq_train[i][k][j]=mse(y_val_train,y_val_pred)
            mean_sq_test[i][k][j]=mse(y_val_test,y_test_pred)
            mean_abs_train[i][k][j]=mabe(y_val_pred,y_val_train)
            mean_abs_test[i][k][j]=mabe(y_test_pred,y_val_test)
            median_abs_train[i][k][j]=mdae(y_val_pred,y_val_train)
            median_abs_test[i][k][j]=mdae(y_val_test,y_test_pred)

            model.fit(x_val_train_lass,y_val_train_lass)

            y_val_pred_lass = model.predict(x_val_train_lass)
            y_test_pred_lass=model.predict(x_val_test_lass)

            R_scores_train_lasso[i][k][j]=r2(y_val_pred_lass,y_val_train_lass)
            R_scores_test_lasso[i][k][j]=r2(y_test_pred_lass,y_val_test_lass)
            mean_sq_train_lasso[i][k][j]=mse(y_val_train_lass,y_val_pred_lass)
            mean_sq_test_lasso[i][k][j]=mse(y_val_test_lass,y_test_pred_lass)
            mean_abs_train_lasso[i][k][j]=mabe(y_val_pred_lass,y_val_train_lass)
            mean_abs_test_lasso[i][k][j]=mabe(y_test_pred_lass,y_val_test_lass)
            median_abs_train_lasso[i][k][j]=mdae(y_val_pred_lass,y_val_train_lass)
            median_abs_tests_lasso[i][k][j]=mdae(y_val_test_lass,y_test_pred_lass)



mean_R_test_lasso=np.mean(R_scores_test_lasso, axis=2)
mean_R_train_lasso=np.mean(R_scores_train_lasso, axis=2)
std_R_test_R_lasso=np.std(R_scores_test_lasso, axis=2)
std_R_train_R_lasso=np.std(R_scores_train_lasso, axis=2)
mean_mean_sq_train_lasso=np.mean(mean_sq_train_lasso, axis=2)
mean_mean_sq_test_lasso=np.mean(mean_sq_test_lasso, axis=2)
std_train_MSE_lasso=np.std(mean_sq_train_lasso, axis=2)
std__test_MSE_lasso=np.std(mean_sq_test_lasso, axis=2)
mean_median_train_lasso=np.mean(median_abs_train_lasso, axis=2)
mean_median_test_lasso=np.mean(median_abs_tests_lasso, axis=2)
mean_mean_abs_train_lasso=np.mean(mean_abs_train_lasso, axis=2)
mean_mean_abs_test_lasso=np.mean(mean_abs_test_lasso,axis=2)


mean_R_test=np.mean(R_scores_test, axis=2)
mean_R_train=np.mean(R_scores_train,axis=2)
std_R_test_R=np.std(R_scores_test, axis=2)
std_R_train_R=np.std(R_scores_train, axis=2)
mean_mean_sq_train=np.mean(mean_sq_train, axis=2)
mean_mean_sq_test=np.mean(mean_sq_test, axis=2)
std_train_MSE=np.std(mean_sq_train, axis=2)
std_test_MSE=np.std(mean_sq_test, axis=2)
mean_median_train=np.mean(median_abs_train, axis=2)
mean_median_test=np.mean(median_abs_test, axis=2)
mean_mean_abs_train=np.mean(mean_abs_train, axis=2)
mean_mean_abs_test=np.mean(mean_abs_test, axis=2)




min_mean_mse_train_index=unravel_index(mean_mean_sq_train.argmin(), mean_mean_sq_train.shape)          
min_mean_mse_test_index=unravel_index(mean_mean_sq_test.argmin(),mean_mean_sq_test.shape)       
max_mean_R2_train_index=unravel_index(mean_mean_sq_test.argmax(),mean_mean_sq_test.shape)       
max_mean_R2_test_index=unravel_index(mean_mean_sq_test.argmax(),mean_mean_sq_test.shape)  
min_median_abs_train_index=unravel_index(mean_median_train.argmin(),mean_median_train.shape)  
min_mean_abs_train_index=unravel_index(mean_mean_abs_train.argmin(),mean_mean_abs_train.shape)  
min_median_abs_test_index=unravel_index(mean_median_test.argmin(),mean_median_test.shape)  
min_mean_abs_test_index=unravel_index(mean_mean_abs_test.argmin(),mean_mean_abs_test.shape)  


min_mean_mse_train_index_lass=unravel_index(mean_mean_sq_train_lasso.argmin(), mean_mean_sq_train_lasso.shape)          
min_mean_mse_test_index_lass=unravel_index(mean_mean_sq_test_lasso.argmin(),mean_mean_sq_test_lasso.shape)       
max_mean_R2_train_index_lass=unravel_index(mean_mean_sq_test_lasso.argmax(),mean_mean_sq_test_lasso.shape)       
max_mean_R2_test_index_lass=unravel_index(mean_mean_sq_test_lasso.argmax(),mean_mean_sq_test_lasso.shape)   
min_median_abs_train_index_lass=unravel_index(mean_median_train_lasso.argmin(),mean_median_train_lasso.shape)  
min_mean_abs_train_index_lass=unravel_index(mean_mean_abs_train_lasso.argmin(),mean_mean_abs_train_lasso.shape)  
min_median_abs_test_index_lass=unravel_index(mean_median_test_lasso.argmin(),mean_median_test_lasso.shape)  
min_mean_abs_test_index_lass=unravel_index(mean_mean_abs_test_lasso.argmin(),mean_mean_abs_test_lasso.shape) 


plt.imshow(mean_mean_sq_test)
plt.colorbar()
plt.show()

#Metrics for RandomForest
print('metrics for Randomforest ridge ')
print("MSE best CV depth  :", depth[min_mean_mse_test_index[0]], "best nestimator: ", estimator[min_mean_mse_test_index[1]], "MSE score:", mean_mean_sq_test[min_mean_mse_test_index[0]][min_mean_mse_test_index[1]])
print("Median Abs best CV depth :", depth[min_median_abs_test_index[0]], "best nestimator :", estimator[min_median_abs_test_index[1]], "Median Abs score:", mean_median_test[min_mean_mse_test_index[0]][min_mean_mse_test_index[1]])
print('R2 best CV depth:', depth[max_mean_R2_test_index[0]], 'with mean R2 score', estimator[max_mean_R2_test_index[1]], 'R2 score :', mean_R_test[min_mean_mse_test_index[0]][min_mean_mse_test_index[1]])
print('Mean abs best CV depth :', depth[min_mean_abs_test_index[0]], 'with mean abs error', estimator[min_mean_abs_test_index[1]], 'Mean abs score', mean_mean_abs_test[min_mean_mse_test_index[0]][min_mean_mse_test_index[1]])

print('metrics for random forest lasso')
print("MSE best CV depth  :", depth[min_mean_mse_test_index_lass[0]], "best nestimator: ", estimator[min_mean_mse_test_index_lass[1]], "MSE score:", mean_mean_sq_test[min_mean_mse_test_index_lass[0]][min_mean_mse_test_index_lass[1]])
print("Median Abs best CV depth :", depth[min_median_abs_test_index_lass[0]], "best nestimator :", estimator[min_median_abs_test_index_lass[1]], "Median Abs score:", mean_median_test[min_mean_mse_test_index_lass[0]][min_mean_mse_test_index_lass[1]])
print('R2 best CV depth:', depth[max_mean_R2_test_index_lass[0]], 'with mean R2 score', estimator[max_mean_R2_test_index_lass[1]], 'R2 score :', mean_R_test[min_mean_mse_test_index_lass[0]][min_mean_mse_test_index_lass[1]])
print('Mean abs best CV depth :', depth[min_mean_abs_test_index_lass[0]], 'with mean abs error', estimator[min_mean_abs_test_index_lass[1]], 'Mean abs score', mean_mean_abs_test_lasso[min_mean_mse_test_index[0]][min_mean_mse_test_index_lass[1]])

#Metrics for RandomForest


##Fitting with the tuned parameters

model = RandomForestRegressor(max_depth=depth[min_mean_mse_test_index[0]],n_estimators=estimator[min_mean_mse_test_index[1]])
model.fit(x_train,y_train)
y_pred = model.predict(x_train)
y_test_pred=model.predict(x_test)

print('MSE Ridge Random Forest train' ,mse(y_pred,y_train))
print('MSE Ridge Random Forest test' ,mse(y_test,y_test_pred))
print('R2 score Ridge Random forest train', r2(y_pred,y_train))
print('R2 score Ridge Random forest test', r2(y_test_pred,y_test))
print('Median abs Ridge Random Forest train' ,mdae(y_pred,y_train))
print('Median abs Ridge Random Forest test' ,mdae(y_test,y_test_pred))
print('Mean abs score Ridge Random forest train', mabe(y_pred,y_train))
print('Mean abs score Ridge Random forest test', mabe(y_test_pred,y_test))

model = RandomForestRegressor(max_depth=depth[min_mean_mse_test_index_lass[0]],n_estimators=estimator[min_mean_mse_test_index_lass[1]])
model.fit(x_train_lass,y_train_lass)
y_pred_lass = model.predict(x_train_lass)
y_test_pred_lass=model.predict(x_test_lass)

print('MSE Lasso Random Forest train' ,mse(y_pred_lass,y_train_lass))
print('MSE Lasso Random Forest test' , mse(y_test_lass,y_test_pred_lass))
print('R2 score Lasso Random forest train', r2(y_pred_lass,y_train_lass))
print('R2 score Lasso Random forest test', r2(y_test_pred_lass,y_test_lass))
print('Median abs Lasso Random Forest train' ,mdae(y_pred_lass,y_train_lass))
print('Median abs Lasso Random Forest test' ,mdae(y_test_lass,y_test_pred_lass))
print('Mean abs score Lasso Random forest train', mabe(y_pred_lass,y_train_lass))
print('Mean abs score Lasso Random forest test', mabe(y_test_pred_lass,y_test_lass))


#Adaboost

R_scores_train_lasso=np.zeros((len(depth),len(estimator),ntrials))
R_scores_test_lasso=np.zeros((len(depth),len(estimator),ntrials))
mean_sq_train_lasso=np.zeros((len(depth),len(estimator),ntrials))
mean_sq_test_lasso=np.zeros((len(depth),len(estimator),ntrials))
mean_abs_test_lasso=np.zeros((len(depth),len(estimator),ntrials))
mean_abs_train_lasso=np.zeros((len(depth),len(estimator),ntrials))
median_abs_train_lasso=np.zeros((len(depth),len(estimator),ntrials))
median_abs_tests_lasso=np.zeros((len(depth),len(estimator),ntrials))


R_scores_train=np.zeros((len(depth),len(estimator),ntrials))
R_scores_test=np.zeros((len(depth),len(estimator),ntrials))
mean_sq_train=np.zeros((len(depth),len(estimator),ntrials))
mean_sq_test=np.zeros((len(depth),len(estimator),ntrials))
mean_abs_train=np.zeros((len(depth),len(estimator),ntrials))
mean_abs_test=np.zeros((len(depth),len(estimator),ntrials))
median_abs_train=np.zeros((len(depth),len(estimator),ntrials))
median_abs_test=np.zeros((len(depth),len(estimator),ntrials))

estimator=[10, 20, 50, 100]

for i in range(len(depth)):
    for k in range(len(estimator)): 
        for j in range(ntrials): 
            x_val_train, x_val_test, y_val_train, y_val_test = train_test_split(x_train, y_train, test_size=0.20) 
            x_val_train_lass, x_val_test_lass, y_val_train_lass, y_val_test_lass = train_test_split(x_train_lass, y_train_lass, test_size=0.20) 
            model = AdaBoostRegressor(DecisionTreeRegressor(max_depth=depth[i]), n_estimators=estimator[k])
            model.fit(x_val_train,y_val_train)
            y_val_pred = model.predict(x_val_train)
            y_test_pred=model.predict(x_val_test)
            R_scores_train[i][k][j]=r2(y_val_pred,y_val_train)
            R_scores_test[i][k][j]=r2(y_test_pred,y_val_test)
            mean_sq_train[i][k][j]=mse(y_val_train,y_val_pred)
            mean_sq_test[i][k][j]=mse(y_val_test,y_test_pred)
            mean_abs_train[i][k][j]=mabe(y_val_pred,y_val_train)
            mean_abs_test[i][k][j]=mabe(y_test_pred,y_val_test)
            median_abs_train[i][k][j]=mdae(y_val_pred,y_val_train)
            median_abs_test[i][k][j]=mdae(y_val_test,y_test_pred)

        
            model.fit(x_val_train_lass,y_val_train_lass)
            y_val_pred_lass = model.predict(x_val_train_lass)
            y_test_pred_lass=model.predict(x_val_test_lass)
            R_scores_train_lasso[i][k][j]=r2(y_val_pred_lass,y_val_train_lass)
            R_scores_test_lasso[i][k][j]=r2(y_test_pred_lass,y_val_test_lass)
            mean_sq_train_lasso[i][k][j]=mse(y_val_train_lass,y_val_pred_lass)
            mean_sq_test_lasso[i][k][j]=mse(y_val_test_lass,y_test_pred_lass)
            mean_abs_train_lasso[i][k][j]=mabe(y_val_pred_lass,y_val_train_lass)
            mean_abs_test_lasso[i][k][j]=mabe(y_test_pred_lass,y_val_test_lass)
            median_abs_train_lasso[i][k][j]=mdae(y_val_pred_lass,y_val_train_lass)
            median_abs_tests_lasso[i][k][j]=mdae(y_val_test_lass,y_test_pred_lass)



mean_R_test_lasso=np.mean(R_scores_test_lasso, axis=2)
mean_R_train_lasso=np.mean(R_scores_train_lasso, axis=2)
std_R_test_R_lasso=np.std(R_scores_test_lasso, axis=2)
std_R_train_R_lasso=np.std(R_scores_train_lasso, axis=2)
mean_mean_sq_train_lasso=np.mean(mean_sq_train_lasso, axis=2)
mean_mean_sq_test_lasso=np.mean(mean_sq_test_lasso, axis=2)
std_train_MSE_lasso=np.std(mean_sq_train_lasso, axis=2)
std__test_MSE_lasso=np.std(mean_sq_test_lasso, axis=2)
mean_median_train_lasso=np.mean(median_abs_train_lasso, axis=2)
mean_median_test_lasso=np.mean(median_abs_tests_lasso, axis=2)
mean_mean_abs_train_lasso=np.mean(mean_abs_train_lasso, axis=2)
mean_mean_abs_test_lasso=np.mean(mean_abs_test_lasso,axis=2)


mean_R_test=np.mean(R_scores_test, axis=2)
mean_R_train=np.mean(R_scores_train,axis=2)
std_R_test_R=np.std(R_scores_test, axis=2)
std_R_train_R=np.std(R_scores_train, axis=2)
mean_mean_sq_train=np.mean(mean_sq_train, axis=2)
mean_mean_sq_test=np.mean(mean_sq_test, axis=2)
std_train_MSE=np.std(mean_sq_train, axis=2)
std_test_MSE=np.std(mean_sq_test, axis=2)
mean_median_train=np.mean(median_abs_train, axis=2)
mean_median_test=np.mean(median_abs_test, axis=2)
mean_mean_abs_train=np.mean(mean_abs_train, axis=2)
mean_mean_abs_test=np.mean(mean_abs_test, axis=2)



min_mean_mse_train_index=unravel_index(mean_mean_sq_train.argmin(), mean_mean_sq_train.shape)          
min_mean_mse_test_index=unravel_index(mean_mean_sq_test.argmin(),mean_mean_sq_test.shape)       
max_mean_R2_train_index=unravel_index(mean_R_train.argmax(),mean_R_train.shape)       
max_mean_R2_test_index=unravel_index(mean_R_test.argmax(),mean_R_train.shape)  

min_median_abs_train_index=unravel_index(mean_median_train.argmin(),mean_median_train.shape)  
min_mean_abs_train_index=unravel_index(mean_mean_abs_train.argmin(),mean_mean_abs_train.shape)  
min_median_abs_test_index=unravel_index(mean_median_test.argmin(),mean_median_test.shape)  
min_mean_abs_test_index=unravel_index(mean_mean_abs_test.argmin(),mean_mean_abs_test.shape)  


min_mean_mse_train_index_lass=unravel_index(mean_mean_sq_train_lasso.argmin(), mean_mean_sq_train_lasso.shape)          
min_mean_mse_test_index_lass=unravel_index(mean_mean_sq_test_lasso.argmin(),mean_mean_sq_test_lasso.shape)       
max_mean_R2_train_index_lass=unravel_index(mean_R_test_lasso.argmax(),mean_R_test_lasso.shape)       
max_mean_R2_test_index_lass=unravel_index(mean_R_test_lasso.argmax(),mean_R_test_lasso.shape)   

min_median_abs_train_index_lass=unravel_index(mean_median_train_lasso.argmin(),mean_median_train_lasso.shape)  
min_mean_abs_train_index_lass=unravel_index(mean_mean_abs_train_lasso.argmin(),mean_mean_abs_train_lasso.shape)  
min_median_abs_test_index_lass=unravel_index(mean_median_test_lasso.argmin(),mean_median_test_lasso.shape)  
min_mean_abs_test_index_lass=unravel_index(mean_mean_abs_test_lasso.argmin(),mean_mean_abs_test_lasso.shape)



#Metrics for Adaboost
print('metrics for Adaboost ridge')
print("MSE best CV depth  :", depth[min_mean_mse_test_index[0]], "best nestimator: ", estimator[min_mean_mse_test_index[1]], "MSE score :", mean_mean_sq_test[min_mean_mse_test_index[0]][min_mean_mse_test_index[1]])
print("Median Abs best CV depth :", depth[min_median_abs_test_index[0]], "best nestimator :", estimator[min_median_abs_test_index[1]], "Median Abs score :", mean_median_test[min_mean_mse_test_index[0]][min_mean_mse_test_index[1]])
print('R2 best CV depth:', depth[max_mean_R2_test_index[0]], 'best estimator :', estimator[max_mean_R2_test_index[1]], 'R2 score :', mean_R_test[min_mean_mse_test_index[0]][min_mean_mse_test_index[1]])
print('Mean abs best CV depth :', depth[min_mean_abs_test_index[0]], 'best estimator :', estimator[min_mean_abs_test_index[1]], 'Mean abs score :', mean_mean_abs_test[min_mean_mse_test_index[0]][min_mean_mse_test_index[1]])

print('metrics for adaboost lasso')
print("MSE best CV depth  :", depth[min_mean_mse_test_index_lass[0]], "best nestimator: ", estimator[min_mean_mse_test_index_lass[1]], "MSE score:", mean_mean_sq_test[min_mean_mse_test_index_lass[0]][min_mean_mse_test_index_lass[1]])
print("Median Abs best CV depth :", depth[min_median_abs_test_index_lass[0]], "best nestimator :", estimator[min_median_abs_test_index_lass[1]], "Median Abs score:", mean_median_test[min_mean_mse_test_index_lass[0]][min_mean_mse_test_index_lass[1]])
print('R2 best CV depth:', depth[max_mean_R2_test_index_lass[0]], 'with mean R2 score', estimator[max_mean_R2_test_index_lass[1]], 'R2 score :', mean_R_test[min_mean_mse_test_index_lass[0]][min_mean_mse_test_index_lass[1]])
print('Mean abs best CV depth :', depth[min_mean_abs_test_index_lass[0]], 'with mean abs error', estimator[min_mean_abs_test_index_lass[1]], 'Mean abs score', mean_mean_abs_test_lasso[min_mean_mse_test_index[0]][min_mean_mse_test_index_lass[1]])






##Fitting with the tuned parameters

model = AdaBoostRegressor(DecisionTreeRegressor(max_depth=depth[min_mean_mse_test_index[0]]),n_estimators=estimator[min_mean_mse_test_index[1]])
model.fit(x_train,y_train)
y_pred = model.predict(x_train)
y_test_pred=model.predict(x_test)

print('MSE Ridge AdaBoost train' ,mse(y_pred,y_train))
print('MSE Ridge AdaBoost test' ,mse(y_test,y_test_pred))
print('R2 score Ridge Adaboost train', r2(y_pred,y_train))
print('R2 score Ridge AdaBoost test', r2(y_test_pred,y_test))
print('Median abs Ridge AdaBoost train' ,mdae(y_pred,y_train))
print('Median abs Ridge AdaBoost test' ,mdae(y_test,y_test_pred))
print('Mean abs score Ridge AdaBoost train', mabe(y_pred,y_train))
print('Mean abs score Ridge AdaBoost test', mabe(y_test_pred,y_test))

model = AdaBoostRegressor(DecisionTreeRegressor(max_depth=depth[min_mean_mse_test_index_lass[0]]),n_estimators=estimator[min_mean_mse_test_index_lass[1]])
model.fit(x_train_lass,y_train_lass)
y_pred_lass = model.predict(x_train_lass)
y_test_pred_lass=model.predict(x_test_lass)

print('MSE Lasso AdaBoost train' ,mse(y_pred_lass,y_train_lass))
print('MSE Lasso AdaBoost test' , mse(y_test_lass,y_test_pred_lass))
print('R2 score Lasso AdaBoost test', r2(y_test_pred_lass,y_test_lass))
print('Median abs Lasso AdaBoost train' ,mdae(y_pred_lass,y_train_lass))
print('Median abs Lasso AdaBoost test' ,mdae(y_test_lass,y_test_pred_lass))
print('Mean abs score Lasso AdaBoost train', mabe(y_pred_lass,y_train_lass))
print('Mean abs score Lasso AdaBoost test', mabe(y_test_pred_lass,y_test_lass))

'''

#Linear Regression
R_scores_train_lasso=np.zeros((ntrials))
R_scores_test_lasso=np.zeros((ntrials))
mean_sq_train_lasso=np.zeros((ntrials))
mean_sq_test_lasso=np.zeros((ntrials))
mean_abs_test_lasso=np.zeros((ntrials))
mean_abs_train_lasso=np.zeros((ntrials))
median_abs_train_lasso=np.zeros((ntrials))
median_abs_tests_lasso=np.zeros((ntrials))


R_scores_train=np.zeros((ntrials))
R_scores_test=np.zeros((ntrials))
mean_sq_train=np.zeros((ntrials))
mean_sq_test=np.zeros((ntrials))
mean_abs_train=np.zeros((ntrials))
mean_abs_test=np.zeros((ntrials))
median_abs_train=np.zeros((ntrials))
median_abs_test=np.zeros((ntrials))



for j in range((ntrials)):
    x_val_train, x_val_test, y_val_train, y_val_test = train_test_split(x_train, y_train, test_size=0.20) 
    x_val_train_lass, x_val_test_lass, y_val_train_lass, y_val_test_lass = train_test_split(x_train_lass, y_train_lass, test_size=0.20) 
    model = LinearRegression()
    model.fit(x_val_train,y_val_train)
    y_val_pred = model.predict(x_val_train)
    y_test_pred=model.predict(x_val_test)
    R_scores_train[j]=r2(y_val_pred,y_val_train)
    R_scores_test[j]=r2(y_test_pred,y_val_test)
    mean_sq_train[j]=mse(y_val_train,y_val_pred)
    mean_sq_test[j]=mse(y_val_test,y_test_pred)
    mean_abs_train[j]=mabe(y_val_train,y_val_pred)
    mean_abs_test[j]=mdae(y_test_pred,y_val_test)
    median_abs_train[j]=mabe(y_val_train,y_val_pred)
    median_abs_test[j]=mdae(y_test_pred,y_val_test)

    model.fit(x_val_train_lass,y_val_train_lass)
    y_val_pred_lass = model.predict(x_val_train_lass)
    y_test_pred_lass=model.predict(x_val_test_lass)
    R_scores_train_lasso[j]=r2(y_val_pred_lass,y_val_train_lass)
    R_scores_test_lasso[j]=r2(y_test_pred_lass,y_val_test_lass)
    mean_sq_train_lasso[j]=mse(y_val_train_lass,y_val_pred_lass)
    mean_sq_test_lasso[j]=mse(y_val_test_lass,y_test_pred_lass)
    mean_abs_test_lasso[j]=mabe(y_val_test_lass,y_test_pred_lass)
    mean_abs_train_lasso[j]=mabe(y_val_pred_lass,y_val_train_lass)
    median_abs_train_lasso[j]=mdae(y_val_pred_lass,y_val_train_lass)
    median_abs_tests_lasso[j]=mdae(y_test_pred_lass,y_val_test_lass)

mean_R_test_lasso=np.mean(R_scores_test_lasso)
mean_R_train_lasso=np.mean(R_scores_train_lasso)
std_R_test_R_lasso=np.std(R_scores_test_lasso)
std_R_train_R_lasso=np.std(R_scores_train_lasso)
mean_mean_sq_train_lasso=np.mean(mean_sq_train_lasso)
mean_mean_sq_test_lasso=np.mean(mean_sq_test_lasso)
std_train_MSE_lasso=np.std(mean_sq_train_lasso)
std__test_MSE_lasso=np.std(mean_sq_test_lasso)
mean_median_train_lasso=np.mean(median_abs_train_lasso)
mean_median_test_lasso=np.mean(median_abs_tests_lasso)
mean_mean_abs_train_lasso=np.mean(mean_abs_train_lasso)
mean_mean_abs_test_lasso=np.mean(mean_abs_test_lasso)


mean_R_test=np.mean(R_scores_test)
mean_R_train=np.mean(R_scores_train)
std_R_test_R=np.std(R_scores_test)
std_R_train_R=np.std(R_scores_train)
mean_mean_sq_train=np.mean(mean_sq_train)
mean_mean_sq_test=np.mean(mean_sq_test)
std_train_MSE=np.std(mean_sq_train)
std_test_MSE=np.std(mean_sq_test)
mean_median_train=np.mean(median_abs_train)
mean_median_test=np.mean(median_abs_test)
mean_mean_abs_train=np.mean(mean_abs_train)
mean_mean_abs_test=np.mean(mean_abs_test)

#Metrics for Linear Reg
print('metrics for Lin reg ridge')
print("MSE score :", mean_mean_sq_test)
print("Median Abs score :",mean_median_test )
print('R2 score :', mean_R_test)
print('Mean abs score :', mean_mean_abs_test)

print('metrics for lin reg lasso')
print("MSE score :", mean_mean_sq_test_lasso)
print("Median Abs score :",mean_median_test_lasso )
print('R2 score :', mean_R_test_lasso)
print('Mean abs score :', mean_mean_abs_test_lasso)


#See how it works for the test set 
model = LinearRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_train)
y_test_pred=model.predict(x_test)

print('MSE Ridge LinearReg train' ,mse(y_pred,y_train))
print('MSE Ridge LinearReg test' ,mse(y_test,y_test_pred))
print('R2 score Ridge LinearReg train', r2(y_pred,y_train))
print('R2 score Ridge LinearReg test', r2(y_test_pred,y_test))
print('Median abs Ridge LinearReg train' ,mdae(y_pred,y_train))
print('Median abs Ridge LinearReg test' ,mdae(y_test,y_test_pred))
print('Mean abs score Ridge LinearReg train', mabe(y_pred,y_train))
print('Mean abs score Ridge LinearReg test', mabe(y_test_pred,y_test))

model = LinearRegression()
model.fit(x_train_lass,y_train_lass)
y_pred_lass = model.predict(x_train_lass)
y_test_pred_lass=model.predict(x_test_lass)

print('MSE Lasso LinearReg train' ,mse(y_pred_lass,y_train_lass))
print('MSE Lasso LinearRer test' , mse(y_test_lass,y_test_pred_lass))
print('R2 score Lasso LinearReg train', r2(y_pred_lass,y_train_lass))
print('R2 score Lasso LinearReg test', r2(y_test_pred_lass,y_test_lass))
print('Median abs Lasso LinearReg train' ,mdae(y_pred_lass,y_train_lass))
print('Median abs Lasso LinearReg test' ,mdae(y_test_lass,y_test_pred_lass))
print('Mean abs score Lasso LinearReg train', mabe(y_pred_lass,y_train_lass))
print('Mean abs score Lasso LinearReg test', mabe(y_test_pred_lass,y_test_lass))


#Gradient Boosting Regressor
R_scores_train_lasso=np.zeros((len(depth),len(estimator),ntrials))
R_scores_test_lasso=np.zeros((len(depth),len(estimator),ntrials))
mean_sq_train_lasso=np.zeros((len(depth),len(estimator),ntrials))
mean_sq_test_lasso=np.zeros((len(depth),len(estimator),ntrials))
mean_abs_test_lasso=np.zeros((len(depth),len(estimator),ntrials))
mean_abs_train_lasso=np.zeros((len(depth),len(estimator),ntrials))
median_abs_train_lasso=np.zeros((len(depth),len(estimator),ntrials))
median_abs_tests_lasso=np.zeros((len(depth),len(estimator),ntrials))


R_scores_train=np.zeros((len(depth),len(estimator),ntrials))
R_scores_test=np.zeros((len(depth),len(estimator),ntrials))
mean_sq_train=np.zeros((len(depth),len(estimator),ntrials))
mean_sq_test=np.zeros((len(depth),len(estimator),ntrials))
mean_abs_train=np.zeros((len(depth),len(estimator),ntrials))
mean_abs_test=np.zeros((len(depth),len(estimator),ntrials))
median_abs_train=np.zeros((len(depth),len(estimator),ntrials))
median_abs_test=np.zeros((len(depth),len(estimator),ntrials))


estimator=[20, 100, 200, 250]


for i in range(len(depth)):
    for k in range(len(estimator)):
        for j in range(ntrials):
            x_val_train, x_val_test, y_val_train, y_val_test = train_test_split(x_train, y_train, test_size=0.20) 
            x_val_train_lass, x_val_test_lass, y_val_train_lass, y_val_test_lass = train_test_split(x_train_lass, y_train_lass, test_size=0.20) 
            model = GradientBoostingRegressor(n_estimators=estimator[k], max_depth=depth[i])
            model.fit(x_val_train, y_val_train)
            y_val_pred = model.predict(x_val_train)
            y_test_pred=model.predict(x_val_test)
            R_scores_train[i][k][j]=r2(y_val_pred,y_val_train)
            R_scores_test[i][k][j]=r2(y_test_pred,y_val_test)
            mean_sq_train[i][k][j]=mse(y_val_train,y_val_pred)
            mean_sq_test[i][k][j]=mse(y_val_test,y_test_pred)
            mean_abs_train[i][k][j]=mabe(y_val_pred,y_val_train)
            mean_abs_test[i][k][j]=mabe(y_test_pred,y_val_test)
            median_abs_train[i][k][j]=mdae(y_val_pred,y_val_train)
            median_abs_test[i][k][j]=mdae(y_val_test,y_test_pred)

            

            model.fit(x_val_train_lass, y_val_train_lass)
            y_val_pred_lass = model.predict(x_val_train_lass)
            y_test_pred_lass=model.predict(x_val_test_lass)
            R_scores_train_lasso[i][k][j]=r2(y_val_pred_lass,y_val_train_lass)
            R_scores_test_lasso[i][k][j]=r2(y_test_pred_lass,y_val_test_lass)
            mean_sq_train[i][k][j]=mse(y_val_train_lass,y_val_pred_lass)
            mean_sq_test_lasso[i][k][j]=mse(y_val_test_lass,y_test_pred_lass)
            mean_abs_train_lasso[i][k][j]=mabe(y_val_pred_lass,y_val_train_lass)
            mean_abs_test_lasso[i][k][j]=mabe(y_test_pred_lass,y_val_test_lass)
            median_abs_train_lasso[i][k][j]=mdae(y_val_pred_lass,y_val_train_lass)
            median_abs_tests_lasso[i][k][j]=mdae(y_val_test_lass,y_test_pred_lass)

                


mean_R_test_lasso=np.mean(R_scores_test_lasso, axis=2)
mean_R_train_lasso=np.mean(R_scores_train_lasso, axis=2)
std_R_test_R_lasso=np.std(R_scores_test_lasso, axis=2)
std_R_train_R_lasso=np.std(R_scores_train_lasso, axis=2)
mean_mean_sq_train_lasso=np.mean(mean_sq_train_lasso, axis=2)
mean_mean_sq_test_lasso=np.mean(mean_sq_test_lasso, axis=2)
std_train_MSE_lasso=np.std(mean_sq_train_lasso, axis=2)
std__test_MSE_lasso=np.std(mean_sq_test_lasso, axis=2)
mean_median_train_lasso=np.mean(median_abs_train_lasso, axis=2)
mean_median_test_lasso=np.mean(median_abs_tests_lasso, axis=2)
mean_mean_abs_train_lasso=np.mean(mean_abs_train_lasso, axis=2)
mean_mean_abs_test_lasso=np.mean(mean_abs_test_lasso,axis=2)


mean_R_test=np.mean(R_scores_test, axis=2)
mean_R_train=np.mean(R_scores_train,axis=2)
std_R_test_R=np.std(R_scores_test, axis=2)
std_R_train_R=np.std(R_scores_train, axis=2)
mean_mean_sq_train=np.mean(mean_sq_train, axis=2)
mean_mean_sq_test=np.mean(mean_sq_test, axis=2)
std_train_MSE=np.std(mean_sq_train, axis=2)
std_test_MSE=np.std(mean_sq_test, axis=2)
mean_median_train=np.mean(median_abs_train, axis=2)
mean_median_test=np.mean(median_abs_test, axis=2)
mean_mean_abs_train=np.mean(mean_abs_train, axis=2)
mean_mean_abs_test=np.mean(mean_abs_test, axis=2)



min_mean_mse_train_index=unravel_index(mean_mean_sq_train.argmin(), mean_mean_sq_train.shape)          
min_mean_mse_test_index=unravel_index(mean_mean_sq_test.argmin(),mean_mean_sq_test.shape)       
max_mean_R2_train_index=unravel_index(mean_R_train.argmax(),mean_R_train.shape)       
max_mean_R2_test_index=unravel_index(mean_R_test.argmax(),mean_R_train.shape)  

min_median_abs_train_index=unravel_index(mean_median_train.argmin(),mean_median_train.shape)  
min_mean_abs_train_index=unravel_index(mean_mean_abs_train.argmin(),mean_mean_abs_train.shape)  
min_median_abs_test_index=unravel_index(mean_median_test.argmin(),mean_median_test.shape)  
min_mean_abs_test_index=unravel_index(mean_mean_abs_test.argmin(),mean_mean_abs_test.shape)  


min_mean_mse_train_index_lass=unravel_index(mean_mean_sq_train_lasso.argmin(), mean_mean_sq_train_lasso.shape)          
min_mean_mse_test_index_lass=unravel_index(mean_mean_sq_test_lasso.argmin(),mean_mean_sq_test_lasso.shape)       
max_mean_R2_train_index_lass=unravel_index(mean_R_test_lasso.argmax(),mean_R_test_lasso.shape)       
max_mean_R2_test_index_lass=unravel_index(mean_R_test_lasso.argmax(),mean_R_test_lasso.shape)   

min_median_abs_train_index_lass=unravel_index(mean_median_train_lasso.argmin(),mean_median_train_lasso.shape)  
min_mean_abs_train_index_lass=unravel_index(mean_mean_abs_train_lasso.argmin(),mean_mean_abs_train_lasso.shape)  
min_median_abs_test_index_lass=unravel_index(mean_median_test_lasso.argmin(),mean_median_test_lasso.shape)  
min_mean_abs_test_index_lass=unravel_index(mean_mean_abs_test_lasso.argmin(),mean_mean_abs_test_lasso.shape) 

#Metrics for GradBoost
print('metrics for gradboost ridge ')
print("MSE best CV depth  :", depth[min_mean_mse_test_index[0]], "best nestimator: ", estimator[min_mean_mse_test_index[1]], "MSE score :", mean_mean_sq_test[min_mean_mse_test_index[0]][min_mean_mse_test_index[1]])
print("Median Abs best CV depth :", depth[min_median_abs_test_index[0]], "best nestimator :", estimator[min_median_abs_test_index[1]], "Median Abs score :", mean_median_test[min_mean_mse_test_index[0]][min_mean_mse_test_index[1]])
print('R2 best CV depth:', depth[max_mean_R2_test_index[0]], 'best estimator :', estimator[max_mean_R2_test_index[1]], 'R2 score :', mean_R_test[min_mean_mse_test_index[0]][min_mean_mse_test_index[1]])
print('Mean abs best CV depth :', depth[min_mean_abs_test_index[0]], 'best estimator :', estimator[min_mean_abs_test_index[1]], 'Mean abs score :', mean_mean_abs_test[min_mean_mse_test_index[0]][min_mean_mse_test_index[1]])

print('metrics for gradboost lasso')
print("MSE best CV depth  :", depth[min_mean_mse_test_index_lass[0]], "best nestimator: ", estimator[min_mean_mse_test_index_lass[1]], "MSE score:", mean_mean_sq_test[min_mean_mse_test_index_lass[0]][min_mean_mse_test_index_lass[1]])
print("Median Abs best CV depth :", depth[min_median_abs_test_index_lass[0]], "best nestimator :", estimator[min_median_abs_test_index_lass[1]], "Median Abs score:", mean_median_test[min_mean_mse_test_index_lass[0]][min_mean_mse_test_index_lass[1]])
print('R2 best CV depth:', depth[max_mean_R2_test_index_lass[0]], 'with mean R2 score', estimator[max_mean_R2_test_index_lass[1]], 'R2 score :', mean_R_test[min_mean_mse_test_index_lass[0]][min_mean_mse_test_index_lass[1]])
print('Mean abs best CV depth :', depth[min_mean_abs_test_index_lass[0]], 'with mean abs error', estimator[min_mean_abs_test_index_lass[1]], 'Mean abs score', mean_mean_abs_test_lasso[min_mean_mse_test_index[0]][min_mean_mse_test_index_lass[1]])


#Metrics for GradBoost

#fit for tuned parameters

model = GradientBoostingRegressor(max_depth=depth[min_mean_mse_test_index[0]],n_estimators=estimator[min_mean_mse_test_index[1]])
model.fit(x_train,y_train)
y_pred = model.predict(x_train)
y_test_pred=model.predict(x_test)

print('MSE Ridge GradBoost train' ,mse(y_pred,y_train))
print('MSE Ridge GradBoost test' ,mse(y_test,y_test_pred))
print('R2 score GradBoost train', r2(y_pred,y_train))
print('R2 score GradBoost test', r2(y_test_pred,y_test))
print('Median abs GradBoost train' ,mdae(y_pred,y_train))
print('Median abs GradBoost test' ,mdae(y_test,y_test_pred))
print('Mean abs score GradBoost train', mabe(y_pred,y_train))
print('Mean abs score GradBoost test', mabe(y_test_pred,y_test))


model = GradientBoostingRegressor(max_depth=depth[min_mean_mse_test_index_lass[0]],n_estimators=estimator[min_mean_mse_test_index_lass[1]])
model.fit(x_train_lass,y_train_lass)
y_pred_lass = model.predict(x_train_lass)
y_test_pred_lass=model.predict(x_test_lass)

print('MSE Lasso GradBoost train' ,mse(y_pred_lass,y_train_lass))
print('MSE Lasso GradBoost test' , mse(y_test_lass,y_test_pred_lass))
print('R2 score Lasso GradBoost train', r2(y_pred_lass,y_train_lass))
print('R2 score Lasso GradBoost test', r2(y_test_pred_lass,y_test_lass))
print('Median abs Lasso GradBoost train' ,mdae(y_pred_lass,y_train_lass))
print('Median abs Lasso GradBoost test' ,mdae(y_test_lass,y_test_pred_lass))
print('Mean abs score Lasso GradBoost train', mabe(y_pred_lass,y_train_lass))
print('Mean abs score Lasso GradBoost test', mabe(y_test_pred_lass,y_test_lass))



n_alphas = 10

alphas = np.logspace(-2, 3, n_alphas)


#Ridge Regression
R_scores_train_lasso=np.zeros((ntrials,n_alphas))
R_scores_test_lasso=np.zeros((ntrials,n_alphas))
mean_sq_train_lasso=np.zeros((ntrials,n_alphas))
mean_sq_test_lasso=np.zeros((ntrials,n_alphas))
mean_abs_test_lasso=np.zeros((ntrials,n_alphas))
mean_abs_train_lasso=np.zeros((ntrials,n_alphas))
median_abs_train_lasso=np.zeros((ntrials,n_alphas))
median_abs_tests_lasso=np.zeros((ntrials,n_alphas))


R_scores_train=np.zeros((ntrials,n_alphas))
R_scores_test=np.zeros((ntrials,n_alphas))
mean_sq_train=np.zeros((ntrials,n_alphas))
mean_sq_test=np.zeros((ntrials,n_alphas))
mean_abs_train=np.zeros((ntrials,n_alphas))
mean_abs_test=np.zeros((ntrials,n_alphas))
median_abs_train=np.zeros((ntrials,n_alphas))
median_abs_test=np.zeros((ntrials,n_alphas))


for j in range((ntrials)):
    for k in range(len(alphas)):
        x_val_train, x_val_test, y_val_train, y_val_test = train_test_split(x_train, y_train, test_size=0.20) 
        x_val_train_lass, x_val_test_lass, y_val_train_lass, y_val_test_lass = train_test_split(x_train_lass, y_train_lass, test_size=0.20) 
        model = linear_model.Ridge(alpha=alphas[k], fit_intercept=True, normalize=True)
        model.fit(x_val_train,y_val_train)
        y_val_pred = model.predict(x_val_train)
        y_test_pred=model.predict(x_val_test)
        R_scores_train[j][k]=r2(y_val_pred,y_val_train)
        R_scores_test[j][k]=r2(y_test_pred,y_val_test)
        mean_sq_train[j][k]=mse(y_val_train,y_val_pred)
        mean_sq_test[j][k]=mse(y_val_test,y_test_pred)
        mean_abs_train[j][k]=mabe(y_val_train,y_val_pred)
        mean_abs_test[j][k]=mdae(y_test_pred,y_val_test)
        median_abs_train[j][k]=mabe(y_val_train,y_val_pred)
        median_abs_test[j][k]=mdae(y_test_pred,y_val_test)

        model.fit(x_val_train_lass,y_val_train_lass)
        y_val_pred_lass = model.predict(x_val_train_lass)
        y_test_pred_lass=model.predict(x_val_test_lass)
        R_scores_train_lasso[j]=r2(y_val_pred_lass,y_val_train_lass)
        R_scores_test_lasso[j]=r2(y_test_pred_lass,y_val_test_lass)
        mean_sq_train_lasso[j]=mse(y_val_train_lass,y_val_pred_lass)
        mean_sq_test_lasso[j]=mse(y_val_test_lass,y_test_pred_lass)
        mean_abs_test_lasso[j]=mabe(y_val_test_lass,y_test_pred_lass)
        mean_abs_train_lasso[j]=mabe(y_val_pred_lass,y_val_train_lass)
        median_abs_train_lasso[j]=mdae(y_val_pred_lass,y_val_train_lass)
        median_abs_tests_lasso[j]=mdae(y_test_pred_lass,y_val_test_lass)

mean_R_test_lasso=np.mean(R_scores_test_lasso, axis=0)
mean_R_train_lasso=np.mean(R_scores_train_lasso, axis=0)
std_R_test_R_lasso=np.std(R_scores_test_lasso, axis=0)
std_R_train_R_lasso=np.std(R_scores_train_lasso, axis=0)
mean_mean_sq_train_lasso=np.mean(mean_sq_train_lasso, axis=0)
mean_mean_sq_test_lasso=np.mean(mean_sq_test_lasso, axis=0)
std_train_MSE_lasso=np.std(mean_sq_train_lasso, axis=0)
std__test_MSE_lasso=np.std(mean_sq_test_lasso, axis=0)
mean_median_train_lasso=np.mean(median_abs_train_lasso, axis=0)
mean_median_test_lasso=np.mean(median_abs_tests_lasso,axis=0)
mean_mean_abs_train_lasso=np.mean(mean_abs_train_lasso, axis=0)
mean_mean_abs_test_lasso=np.mean(mean_abs_test_lasso, axis=0)


mean_R_test=np.mean(R_scores_test, axis=0)
mean_R_train=np.mean(R_scores_train, axis=0)
std_R_test_R=np.std(R_scores_test, axis=0)
std_R_train_R=np.std(R_scores_train, axis=0)
mean_mean_sq_train=np.mean(mean_sq_train, axis=0)
mean_mean_sq_test=np.mean(mean_sq_test, axis=0)
std_train_MSE=np.std(mean_sq_train, axis=0)
std_test_MSE=np.std(mean_sq_test, axis=0)
mean_median_train=np.mean(median_abs_train, axis=0)
mean_median_test=np.mean(median_abs_test, axis=0)
mean_mean_abs_train=np.mean(mean_abs_train, axis=0)
mean_mean_abs_test=np.mean(mean_abs_test, axis=0)



min_mean_mse_train_index=unravel_index(mean_mean_sq_train.argmin(), mean_mean_sq_train.shape)          
min_mean_mse_test_index=unravel_index(mean_mean_sq_test.argmin(),mean_mean_sq_test.shape)       
max_mean_R2_train_index=unravel_index(mean_R_train.argmax(),mean_R_train.shape)       
max_mean_R2_test_index=unravel_index(mean_R_test.argmax(),mean_R_train.shape)  

min_median_abs_train_index=unravel_index(mean_median_train.argmin(),mean_median_train.shape)  
min_mean_abs_train_index=unravel_index(mean_mean_abs_train.argmin(),mean_mean_abs_train.shape)  
min_median_abs_test_index=unravel_index(mean_median_test.argmin(),mean_median_test.shape)  
min_mean_abs_test_index=unravel_index(mean_mean_abs_test.argmin(),mean_mean_abs_test.shape)  


min_mean_mse_train_index_lass=unravel_index(mean_mean_sq_train_lasso.argmin(), mean_mean_sq_train_lasso.shape)          
min_mean_mse_test_index_lass=unravel_index(mean_mean_sq_test_lasso.argmin(),mean_mean_sq_test_lasso.shape)       
max_mean_R2_train_index_lass=unravel_index(mean_R_test_lasso.argmax(),mean_R_test_lasso.shape)       
max_mean_R2_test_index_lass=unravel_index(mean_R_test_lasso.argmax(),mean_R_test_lasso.shape)   

min_median_abs_train_index_lass=unravel_index(mean_median_train_lasso.argmin(),mean_median_train_lasso.shape)  
min_mean_abs_train_index_lass=unravel_index(mean_mean_abs_train_lasso.argmin(),mean_mean_abs_train_lasso.shape)  
min_median_abs_test_index_lass=unravel_index(mean_median_test_lasso.argmin(),mean_median_test_lasso.shape)  
min_mean_abs_test_index_lass=unravel_index(mean_mean_abs_test_lasso.argmin(),mean_mean_abs_test_lasso.shape) 



#Metrics for Ridge

print('metrics for ridge ridge')
print("MSE score  :", mean_mean_sq_test[min_mean_mse_test_index], 'with alpha :', alphas[min_mean_mse_test_index])
print("Median Abs score :", mean_median_test[min_median_abs_test_index], 'with alpha : ', alphas[min_median_abs_test_index])
print('R2 score :', mean_R_test[max_mean_R2_test_index], 'with alpha :', alphas[max_mean_R2_test_index])
print('Mean abs score :', mean_mean_abs_test[min_mean_abs_test_index], 'with alpha :', alphas[min_mean_abs_test_index])

print('metrics for ridge lasso')
print("MSE score  :", mean_mean_sq_test_lasso[min_mean_mse_test_index_lass], alphas[min_mean_mse_test_index_lass])
print("Median Abs score :", mean_median_test_lasso[min_median_abs_test_index_lass],alphas[min_median_abs_test_index_lass] )
print('R2 score :', mean_R_test_lasso[max_mean_R2_test_index_lass], alphas[max_mean_R2_test_index_lass])
print('Mean abs score :', mean_mean_abs_test_lasso[min_mean_abs_test_index_lass],'with alpha :', alphas[min_mean_abs_test_index_lass])

#Fitting for the tuned parameters
model = linear_model.Ridge(alpha=alphas[min_mean_mse_test_index],normalize=True)
model.fit(x_train,y_train)
y_pred = model.predict(x_train)
y_test_pred=model.predict(x_test)

print('MSE Ridge Ridge train' ,mse(y_pred,y_train))
print('MSE Ridge Ridge test' ,mse(y_test,y_test_pred))
print('R2 score Ridge train', r2(y_pred,y_train))
print('R2 score Ridge test', r2(y_test_pred,y_test))
print('Median abs Ridge train' ,mdae(y_pred,y_train))
print('Median abs Ridge test' ,mdae(y_test,y_test_pred))
print('Mean abs score Ridge train', mabe(y_pred,y_train))
print('Mean abs score Ridge test', mabe(y_test_pred,y_test))


model = linear_model.Ridge(alpha=alphas[min_mean_abs_test_index_lass], normalize=True)
model.fit(x_train_lass,y_train_lass)
y_pred_lass = model.predict(x_train_lass)
y_test_pred_lass=model.predict(x_test_lass)

print('MSE Lasso Ridge train' ,mse(y_pred_lass,y_train_lass))
print('MSE Lasso Ridge test' , mse(y_test_lass,y_test_pred_lass))
print('R2 score Lasso Rideg train', r2(y_pred_lass,y_train_lass))
print('R2 score Lasso Ridge test', r2(y_test_pred_lass,y_test_lass))
print('Median abs Lasso Ridge train' ,mdae(y_pred_lass,y_train_lass))
print('Median abs Lasso Ridge test' ,mdae(y_test_lass,y_test_pred_lass))
print('Mean abs score Lasso Ridge train', mabe(y_pred_lass,y_train_lass))
print('Mean abs score Lasso Ridge test', mabe(y_test_pred_lass,y_test_lass))



#Lasso Regression
n_alphas = 10
alphas = np.logspace(-2, 3, n_alphas)
R_scores_train_lasso=np.zeros((ntrials,n_alphas))
R_scores_test_lasso=np.zeros((ntrials,n_alphas))
mean_sq_train_lasso=np.zeros((ntrials,n_alphas))
mean_sq_test_lasso=np.zeros((ntrials,n_alphas))
mean_abs_test_lasso=np.zeros((ntrials,n_alphas))
mean_abs_train_lasso=np.zeros((ntrials,n_alphas))
median_abs_train_lasso=np.zeros((ntrials,n_alphas))
median_abs_tests_lasso=np.zeros((ntrials,n_alphas))


R_scores_train=np.zeros((ntrials,n_alphas))
R_scores_test=np.zeros((ntrials,n_alphas))
mean_sq_train=np.zeros((ntrials,n_alphas))
mean_sq_test=np.zeros((ntrials,n_alphas))
mean_abs_train=np.zeros((ntrials,n_alphas))
mean_abs_test=np.zeros((ntrials,n_alphas))
median_abs_train=np.zeros((ntrials,n_alphas))
median_abs_test=np.zeros((ntrials,n_alphas))


for j in range((ntrials)):
    for k in range(len(alphas)):
        x_val_train, x_val_test, y_val_train, y_val_test = train_test_split(x_train, y_train, test_size=0.20) 
        x_val_train_lass, x_val_test_lass, y_val_train_lass, y_val_test_lass = train_test_split(x_train_lass, y_train_lass, test_size=0.20) 
        model = linear_model.Lasso(alpha=alphas[k], normalize=True)
        model.fit(x_val_train,y_val_train)
        y_val_pred = model.predict(x_val_train)
        y_test_pred=model.predict(x_val_test)
        R_scores_train[j][k]=r2(y_val_pred,y_val_train)
        R_scores_test[j][k]=r2(y_test_pred,y_val_test)
        mean_sq_train[j][k]=mse(y_val_train,y_val_pred)
        mean_sq_test[j][k]=mse(y_val_test,y_test_pred)
        mean_abs_train[j][k]=mabe(y_val_train,y_val_pred)
        mean_abs_test[j][k]=mdae(y_test_pred,y_val_test)
        median_abs_train[j][k]=mabe(y_val_train,y_val_pred)
        median_abs_test[j][k]=mdae(y_test_pred,y_val_test)

        model.fit(x_val_train_lass,y_val_train_lass)
        y_val_pred_lass = model.predict(x_val_train_lass)
        y_test_pred_lass=model.predict(x_val_test_lass)
        R_scores_train_lasso[j]=r2(y_val_pred_lass,y_val_train_lass)
        R_scores_test_lasso[j]=r2(y_test_pred_lass,y_val_test_lass)
        mean_sq_train_lasso[j]=mse(y_val_train_lass,y_val_pred_lass)
        mean_sq_test_lasso[j]=mse(y_val_test_lass,y_test_pred_lass)
        mean_abs_test_lasso[j]=mabe(y_val_test_lass,y_test_pred_lass)
        mean_abs_train_lasso[j]=mabe(y_val_pred_lass,y_val_train_lass)
        median_abs_train_lasso[j]=mdae(y_val_pred_lass,y_val_train_lass)
        median_abs_tests_lasso[j]=mdae(y_test_pred_lass,y_val_test_lass)

mean_R_test_lasso=np.mean(R_scores_test_lasso, axis=0)
mean_R_train_lasso=np.mean(R_scores_train_lasso, axis=0)
std_R_test_R_lasso=np.std(R_scores_test_lasso, axis=0)
std_R_train_R_lasso=np.std(R_scores_train_lasso, axis=0)
mean_mean_sq_train_lasso=np.mean(mean_sq_train_lasso, axis=0)
mean_mean_sq_test_lasso=np.mean(mean_sq_test_lasso, axis=0)
std_train_MSE_lasso=np.std(mean_sq_train_lasso, axis=0)
std__test_MSE_lasso=np.std(mean_sq_test_lasso, axis=0)
mean_median_train_lasso=np.mean(median_abs_train_lasso, axis=0)
mean_median_test_lasso=np.mean(median_abs_tests_lasso,axis=0)
mean_mean_abs_train_lasso=np.mean(mean_abs_train_lasso, axis=0)
mean_mean_abs_test_lasso=np.mean(mean_abs_test_lasso, axis=0)


mean_R_test=np.mean(R_scores_test, axis=0)
mean_R_train=np.mean(R_scores_train, axis=0)
std_R_test_R=np.std(R_scores_test, axis=0)
std_R_train_R=np.std(R_scores_train, axis=0)
mean_mean_sq_train=np.mean(mean_sq_train, axis=0)
mean_mean_sq_test=np.mean(mean_sq_test, axis=0)
std_train_MSE=np.std(mean_sq_train, axis=0)
std_test_MSE=np.std(mean_sq_test, axis=0)
mean_median_train=np.mean(median_abs_train, axis=0)
mean_median_test=np.mean(median_abs_test, axis=0)
mean_mean_abs_train=np.mean(mean_abs_train, axis=0)
mean_mean_abs_test=np.mean(mean_abs_test, axis=0)

min_mean_mse_train_index=unravel_index(mean_mean_sq_train.argmin(), mean_mean_sq_train.shape)          
min_mean_mse_test_index=unravel_index(mean_mean_sq_test.argmin(),mean_mean_sq_test.shape)       
max_mean_R2_train_index=unravel_index(mean_R_train.argmax(),mean_R_train.shape)       
max_mean_R2_test_index=unravel_index(mean_R_test.argmax(),mean_R_train.shape)  

min_median_abs_train_index=unravel_index(mean_median_train.argmin(),mean_median_train.shape)  
min_mean_abs_train_index=unravel_index(mean_mean_abs_train.argmin(),mean_mean_abs_train.shape)  
min_median_abs_test_index=unravel_index(mean_median_test.argmin(),mean_median_test.shape)  
min_mean_abs_test_index=unravel_index(mean_mean_abs_test.argmin(),mean_mean_abs_test.shape)  


min_mean_mse_train_index_lass=unravel_index(mean_mean_sq_train_lasso.argmin(), mean_mean_sq_train_lasso.shape)          
min_mean_mse_test_index_lass=unravel_index(mean_mean_sq_test_lasso.argmin(),mean_mean_sq_test_lasso.shape)       
max_mean_R2_train_index_lass=unravel_index(mean_R_test_lasso.argmax(),mean_R_test_lasso.shape)       
max_mean_R2_test_index_lass=unravel_index(mean_R_test_lasso.argmax(),mean_R_test_lasso.shape)   

min_median_abs_train_index_lass=unravel_index(mean_median_train_lasso.argmin(),mean_median_train_lasso.shape)  
min_mean_abs_train_index_lass=unravel_index(mean_mean_abs_train_lasso.argmin(),mean_mean_abs_train_lasso.shape)  
min_median_abs_test_index_lass=unravel_index(mean_median_test_lasso.argmin(),mean_median_test_lasso.shape)  
min_mean_abs_test_index_lass=unravel_index(mean_mean_abs_test_lasso.argmin(),mean_mean_abs_test_lasso.shape) 



#Metrics for Lasso

print('metrics for ridge Lasso')
print("MSE score  :", mean_mean_sq_test[min_mean_mse_test_index], 'with alpha :', alphas[min_mean_mse_test_index])
print("Median Abs score :", mean_median_test[min_median_abs_test_index], 'with alpha : ', alphas[min_median_abs_test_index])
print('R2 score :', mean_R_test[max_mean_R2_test_index], 'with alpha :', alphas[max_mean_R2_test_index])
print('Mean abs score :', mean_mean_abs_test[min_mean_abs_test_index], 'with alpha :', alphas[min_mean_abs_test_index])

print('metrics for ridge lasso')
print("MSE score  :", mean_mean_sq_test_lasso[min_mean_mse_test_index_lass], alphas[min_mean_mse_test_index_lass])
print("Median Abs score :", mean_median_test_lasso[min_median_abs_test_index_lass],alphas[min_median_abs_test_index_lass] )
print('R2 score :', mean_R_test_lasso[max_mean_R2_test_index_lass], alphas[max_mean_R2_test_index_lass])
print('Mean abs score :', mean_mean_abs_test_lasso[min_mean_abs_test_index_lass],'with alpha :', alphas[min_mean_abs_test_index_lass])

#Fitting for the tuned parameters
model = linear_model.Lasso(alpha=alphas[min_mean_mse_test_index], normalize=True)
model.fit(x_train,y_train)
y_pred = model.predict(x_train)
y_test_pred=model.predict(x_test)

print('MSE Ridge Lasso train' ,mse(y_pred,y_train))
print('MSE Ridge Lasso test' ,mse(y_test,y_test_pred))
print('R2 score Ridge Lasso train', r2(y_pred,y_train))
print('R2 score Ridge Lasso test', r2(y_test_pred,y_test))
print('Median abs Ridge Lasso train' ,mdae(y_pred,y_train))
print('Median abs Ridge Lasso test' ,mdae(y_test,y_test_pred))
print('Mean abs score Ridge Lasso train', mabe(y_pred,y_train))
print('Mean abs score Ridge Lasso test', mabe(y_test_pred,y_test))


model = linear_model.Lasso(alpha=alphas[min_mean_abs_test_index_lass],normalize=True)
model.fit(x_train_lass,y_train_lass)
y_pred_lass = model.predict(x_train_lass)
y_test_pred_lass=model.predict(x_test_lass)

print('MSE Lasso Lasso train' ,mse(y_pred_lass,y_train_lass))
print('MSE Lasso Lasso test' , mse(y_test_lass,y_test_pred_lass))
print('R2 score Lasso Lasso train', r2(y_pred_lass,y_train_lass))
print('R2 score Lasso Lasso test', r2(y_test_pred_lass,y_test_lass))
print('Median abs Lasso Lasso train' ,mdae(y_pred_lass,y_train_lass))
print('Median abs Lasso Lasso test' ,mdae(y_test_lass,y_test_pred_lass))
print('Mean abs score Lasso Lasso train', mabe(y_pred_lass,y_train_lass))
print('Mean abs score Lasso Lasso test', mabe(y_test_pred_lass,y_test_lass))

