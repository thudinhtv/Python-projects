#Read in Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn import metrics

#Neural Network
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

#Multiple Regression
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as smf
from numpy import log

# AIC, BIC calculation (pip install RegscorePy)
from RegscorePy import *

# Setup the Working Directory
os.chdir('Path\\TimeSeriesData')
os.getcwd()

# Read in data
financial_data = pd.read_table('CaliforniaHospital_FinancialData.txt')
financial_data.shape
financial_data.columns
financial_data.dtypes
financial_data.head()

# Variable of choice for analysis: NONOP_REV
# Make a copy of data for NONOP_REV variable
s1 = financial_data.NONOP_REV
N = s1.size

#1. ARMA(1,1): 
## Create Lag 1 & 4 Effects
lag1col = pd.Series([0])
lag1col = lag1col.append(s1, ignore_index=True)
lag4col = pd.Series([0,0,0,0])
lag4col = lag4col.append(s1, ignore_index=True)

## Make lag cols have same size with the original dataframe
lag1col = lag1col.iloc[0:N]  
lag4col = lag4col.iloc[0:N]

## Create new columns in the data for lag effects 
newcols1 = pd.DataFrame({'lag1': lag1col})
newcols4 = pd.DataFrame({'lag4': lag4col})
financial_data1 = pd.concat([financial_data, newcols1, newcols4], axis=1)
financial_data2 = financial_data1[['NONOP_REV','lag1','lag4']]

## Create Time Variable
timelen = len(financial_data2.index) + 1
newcols3 = pd.DataFrame({'time': list(range(1,timelen))})
financial_data3 = pd.concat([financial_data2, newcols3], axis=1)

##Finalize data with 1 & 4 lag effects
financial_data4 = financial_data3[['NONOP_REV','time','lag1', 'lag4']]

## Creating training and testing data
splitnum = np.round((len(financial_data4.index) * 0.7), 0).astype(int)
financial_data_train = financial_data4.iloc[0:splitnum]
financial_data_test = financial_data4.iloc[splitnum+1:N]

## Create 3 ANN models for ARMA(1,1)
### ANN1: tanh/sgd/hidden_layer_sizes:(100,)
nnts1 = MLPRegressor(activation='tanh', solver='sgd', hidden_layer_sizes=(100,))
nnts1.fit(financial_data_train[['time','lag1','lag4']],financial_data_train.NONOP_REV)
nnts1

### Predicting
nnts1_pred = nnts1.predict(financial_data_test[['time','lag1','lag4']])

### Evaluating
#### MAE (Mean Absolute Error)
mae1 = metrics.mean_absolute_error(financial_data_test.NONOP_REV, nnts1_pred)
print('MAE1:', mae1)

#### MSE (Mean Square Error)
mse1 = metrics.mean_squared_error(financial_data_test.NONOP_REV, nnts1_pred)
print('MSE1:', mse1)

#### MAPE (Mean Absolute Percentage Error)
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape1= mean_absolute_percentage_error(financial_data_test.NONOP_REV, nnts1_pred)
print('MAPE1:', mape1)

### ANN2 logistic/sgd/hidden_layer_sizes:(100,)
nnts2 = MLPRegressor(activation='logistic', solver='sgd', hidden_layer_sizes=(100,))
nnts2.fit(financial_data_train[['time','lag1','lag4']],financial_data_train.NONOP_REV)

### Predicting
nnts2_pred = nnts2.predict(financial_data_test[['time','lag1','lag4']])

### Evaluating
mae2 = metrics.mean_absolute_error(financial_data_test.NONOP_REV, nnts2_pred)
print('MAE2:', mae2)
mse2 = metrics.mean_squared_error(financial_data_test.NONOP_REV, nnts2_pred)
print('MSE2:', mse2)
mape2= mean_absolute_percentage_error(financial_data_test.NONOP_REV, nnts2_pred)
print('MAPE2:', mape2)

### ANN3 logistic/sgd/hidden_layer_sizes:(50,)
nnts3 = MLPRegressor(activation='logistic', solver='sgd', hidden_layer_sizes=(50,))
nnts3.fit(financial_data_train[['time','lag1','lag4']],financial_data_train.NONOP_REV)

### Predicting
nnts3_pred = nnts3.predict(financial_data_test[['time','lag1','lag4']])

### Evaluating
mae3 = metrics.mean_absolute_error(financial_data_test.NONOP_REV, nnts3_pred)
print('MAE3:', mae3)
mse3 = metrics.mean_squared_error(financial_data_test.NONOP_REV, nnts3_pred)
print('MSE3:', mse3)
mape3= mean_absolute_percentage_error(financial_data_test.NONOP_REV, nnts3_pred)
print('MAPE3:', mape3)



#2.ARMA(0,1) 
## Create Lag 1 & 2 Effects
lag1col = pd.Series([0])
lag1col = lag1col.append(s1, ignore_index=True)
lag2col = pd.Series([0,0])
lag2col = lag2col.append(s1, ignore_index=True)

## Make lag cols have same size with the original dataframe
lag1col = lag1col.iloc[0:N]  
lag2col = lag2col.iloc[0:N]

## Create new columns in the data for lag effects 
newcols1 = pd.DataFrame({'lag1': lag1col})
newcols2 = pd.DataFrame({'lag2': lag2col})
financial_data5 = pd.concat([financial_data, newcols1, newcols2], axis=1)
financial_data6 = financial_data5[['NONOP_REV','lag1','lag2']]

## Create Time Variable
timelen = len(financial_data6.index) + 1
newcols3 = pd.DataFrame({'time': list(range(1,timelen))})
financial_data7 = pd.concat([financial_data6, newcols3], axis=1)

## Finalize data with 1 & 2 lag effects
financial_data8 = financial_data7[['NONOP_REV','time','lag1', 'lag2']]

## Creating training and testing data
splitnum = np.round((len(financial_data5.index) * 0.7), 0).astype(int)
financial_data_train2 = financial_data8.iloc[0:splitnum]
financial_data_test2 = financial_data8.iloc[splitnum+1:N]

## Create 3 ANN models for ARMA(0,1)
### ANN4: tanh/sgd/hidden_layer_sizes:(100,100)
nnts4 = MLPRegressor(activation='tanh', solver='sgd', hidden_layer_sizes=(100,))
nnts4.fit(financial_data_train2[['time','lag1','lag2']],financial_data_train2.NONOP_REV)

### Predicting
nnts4_pred = nnts4.predict(financial_data_test2[['time','lag1','lag2']])

### Evaluating
mae4 = metrics.mean_absolute_error(financial_data_test2.NONOP_REV, nnts4_pred)
print('MAE4:', mae4)
mse4 = metrics.mean_squared_error(financial_data_test2.NONOP_REV, nnts4_pred)
print('MSE4:', mse4)
mape4= mean_absolute_percentage_error(financial_data_test2.NONOP_REV, nnts4_pred)
print('MAPE4:', mape4)

### ANN5: logistic/sgd/hidden_layer_sizes:(100,)
nnts5 = MLPRegressor(activation='logistic', solver='sgd', hidden_layer_sizes=(100,))
nnts5.fit(financial_data_train2[['time','lag1','lag2']],financial_data_train2.NONOP_REV)

### Predicting
nnts5_pred = nnts5.predict(financial_data_test2[['time','lag1','lag2']])

### Evaluating
mae5 = metrics.mean_absolute_error(financial_data_test2.NONOP_REV, nnts5_pred)
print('MAE5:', mae5)
mse5 = metrics.mean_squared_error(financial_data_test2.NONOP_REV, nnts5_pred)
print('MSE5:', mse5)
mape5= mean_absolute_percentage_error(financial_data_test2.NONOP_REV, nnts5_pred)
print('MAPE5:', mape5)

### ANN6: logistic/sgd/hidden_layer_sizes:(50,)
nnts6 = MLPRegressor(activation='logistic', solver='sgd', hidden_layer_sizes=(50,))
nnts6.fit(financial_data_train2[['time','lag1','lag2']],financial_data_train2.NONOP_REV)

### Predicting
nnts6_pred = nnts6.predict(financial_data_test2[['time','lag1','lag2']])

### Evaluating
mae6 = metrics.mean_absolute_error(financial_data_test2.NONOP_REV, nnts6_pred)
print('MAE6:', mae6)
mse6 = metrics.mean_squared_error(financial_data_test2.NONOP_REV, nnts6_pred)
print('MSE6:', mse6)
mape6= mean_absolute_percentage_error(financial_data_test2.NONOP_REV, nnts6_pred)
print('MAPE6:', mape6)



## Compare the results among all 6 ANN models

### MAE
fig, ax = plt.subplots()
fig.set_size_inches(14, 8, forward=True)
x = [1, 2, 3, 4, 5, 6]
plt.bar(x, [mae1, mae2, mae3, mae4, mae5, mae6])
plt.ylabel("Mean Absolute Error", fontsize=14)
plt.xlabel("ANN Models", fontsize=14)
plt.xticks(x, ('ANN1\n[lag1,4]\ntanh/sgd\nHidden_Size(100,)',
               'ANN2\n[lag1,4]\nlogistic/sgd\nHidden_Size(100,)',
               'ANN3\n[lag1,4]\nlogistic/sgd\nHidden_Size(50,)',
               'ANN4\n[lag1,2]\ntanh/sgd\nHidden_Size(100,)',
               'ANN5\n[lag1,2]\nlogistic/sgd\nHidden_Size(100,)',
               'ANN6\n[lag1,2]\nlogistic/sgd\nHidden_Size(50,)'),fontsize=12)
plt.show()


### MSE
fig, ax = plt.subplots()
fig.set_size_inches(14, 8, forward=True)
x = [1, 2, 3, 4, 5, 6]
plt.bar(x, [mse1, mse2, mse3, mse4, mse5, mse6])
plt.ylabel("Mean Squared Error", fontsize=14)
plt.xlabel("ANN Models", fontsize=14)
plt.xticks(x, ('ANN1\n[lag1,4]\ntanh/sgd\nHidden_Size(100,)',
               'ANN2\n[lag1,4]\nlogistic/sgd\nHidden_Size(100,)',
               'ANN3\n[lag1,4]\nlogistic/sgd\nHidden_Size(50,)',
               'ANN4\n[lag1,2]\ntanh/sgd\nHidden_Size(100,)',
               'ANN5\n[lag1,2]\nlogistic/sgd\nHidden_Size(100,)',
               'ANN6\n[lag1,2]\nlogistic/sgd\nHidden_Size(50,)'),fontsize=12)
plt.show()


### MAPE
fig, ax = plt.subplots()
fig.set_size_inches(14, 8, forward=True)
x = [1, 2, 3, 4, 5, 6]
plt.bar(x, [mape1, mape2, mape3, mape4, mape5, mape6])
plt.ylabel("Mean Absolute Percentage Error [%]", fontsize=14)
plt.xlabel("ANN Models", fontsize=14)
plt.xticks(x, ('ANN1\n[lag1,4]\ntanh/sgd\nHidden_Size(100,)',
               'ANN2\n[lag1,4]\nlogistic/sgd\nHidden_Size(100,)',
               'ANN3\n[lag1,4]\nlogistic/sgd\nHidden_Size(50,)',
               'ANN4\n[lag1,2]\ntanh/sgd\nHidden_Size(100,)',
               'ANN5\n[lag1,2]\nlogistic/sgd\nHidden_Size(100,)',
               'ANN6\n[lag1,2]\nlogistic/sgd\nHidden_Size(50,)'),fontsize=12)
plt.show()


### AIC, BIC of ANN3
p = 3   #Number of variables (3 variables: time, lag1, lag4)
financial_data_pred = nnts3.predict(financial_data4[['time','lag1','lag4']])

aic.aic(financial_data4.NONOP_REV, financial_data_pred, p)
bic.bic(financial_data4.NONOP_REV, financial_data_pred, p)