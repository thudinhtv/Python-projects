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

# Tree
from sklearn.feature_extraction.image import grid_to_graph
from sklearn import tree
from sklearn import metrics

from sklearn.externals.six import StringIO
from IPython.display import Image, display
import pydotplus


# Setup the Working Directory
os.chdir('Path\\Data')
os.getcwd()


#1. ANN Regression: Using the reduction_data_new.txt datase
# Read in data
reduc_data = pd.read_table('reduction_data_new.txt')
reduc_data.columns
reduc_data.dtypes

# Subset data for regression analysis (using intent01 as target variable
# and only 5 of the other numerical variables as independent variables
data=reduc_data[['peruse02', 'peruse04', 'peruse05', 'pereou03', 
                 'pereou04', 'intent01']]
data=data.astype(float)
data.dtypes

# Split data into training and testing
reduc_data_train, reduc_data_test, y_train, y_test = train_test_split(
        data.iloc[:,1:5], data.intent01, test_size=0.3) 

# Standardize the scaling of the variables by
# computing the mean and std to be used for later scaling.
scaler = preprocessing.StandardScaler()
scaler.fit(reduc_data_train)

# Perform the standardization process
reduc_data_train = scaler.transform(reduc_data_train)
reduc_data_test = scaler.transform(reduc_data_test)

# Create a ANN regession model Neural Network using a logistic function 
# and (20, 20) nodes/layer

nnreg1 = MLPRegressor(activation='logistic', solver='sgd', 
                      hidden_layer_sizes=(20,20), 
                      early_stopping=True)

nnreg1.fit(reduc_data_train, y_train)
nnpred1 = nnreg1.predict(reduc_data_test)
nnreg1.n_layers_
nnreg1.coefs_

# Assess the model and results
metrics.mean_absolute_error(y_test, nnpred1)
mse1 = metrics.mean_squared_error(y_test, nnpred1)
print("Mean Squared Error:", mse1)
metrics.r2_score(y_test, nnpred1)

# Create two more models to improve the accuracy error.
## a.Neural Network using rectified linear unit function, (20, 20) nodes/layer
nnreg2 = MLPRegressor(activation='relu', solver='sgd',
                      hidden_layer_sizes=(20,20), 
                       early_stopping=True)
nnreg2.fit(reduc_data_train, y_train)
nnpred2 = nnreg2.predict(reduc_data_test)
metrics.mean_absolute_error(y_test, nnpred2)
mse2 = metrics.mean_squared_error(y_test, nnpred2)
print("Mean Squared Error:", mse2)
metrics.r2_score(y_test, nnpred2)

## b.Neural Network: (100, 100) nodes/layer and no early stopping
nnreg3 = MLPRegressor(activation='relu', solver='sgd', 
                      hidden_layer_sizes=(100,100),
                      early_stopping=True)

nnreg3.fit(reduc_data_train, y_train)
nnpred3 = nnreg3.predict(reduc_data_test)

metrics.mean_absolute_error(y_test, nnpred3)
mse3 = metrics.mean_squared_error(y_test, nnpred3)
print("Mean Squared Error:", mse3)
metrics.r2_score(y_test, nnpred3)

nnreg3.hidden_layer_sizes        #Number of nodes per layer
#### Result: 100 nodes in two layers

## c.Linear Regessition model
linreg = LinearRegression(fit_intercept=True, normalize=True)
linreg.fit(reduc_data_train, y_train)
linreg.score(reduc_data_test, y_test)
VIF = 1/(1-linreg.score(reduc_data_test, y_test))
print(VIF)

### Compute mean_square_error
y_predict = linreg.predict(reduc_data_test)
N = len(y_test)
mse4 = np.sum((np.array(y_test).flatten() - np.array(y_predict).flatten())**2)/N
print("RMSE test: ", mse4) 

## Compare the results against a linear regression model (run all together)
fig, ax = plt.subplots()
x = [1,2,3,4]
plt.bar(x, [mse1, mse2, mse3, mse4])
plt.ylabel("Mean Square Error", fontsize=12)
plt.xticks(x, ('ANN1\nlogistic,sgd\n(20,20)', 'ANN2\nrelu,sgd\n(20,20)', 
               'ANN3\nrelu,sgd\n(100,100)', 'Linear Reg.'))
plt.show()


# 2. ANN Classification: Use the titanic_data.txt file 
# Read in data
titanic_data = pd.read_csv('titanic_data.txt', sep='\t')
titanic_data.columns
titanic_data.dtypes

# Convert to numeric variables
titanic_data.Class.unique()
titanic_data.Class.replace(['1st', '2nd', '3rd', 'Crew'], [0, 1, 2, 3], inplace=True)

titanic_data.Sex.unique()
titanic_data.Sex.replace(['Male', 'Female'], [0, 1], inplace=True)

titanic_data.Age.unique()
titanic_data.Age.replace(['Child', 'Adult'], [0, 1], inplace=True)

titanic_data.Survived.unique()
titanic_data.Survived.replace(['No', 'Yes'], [0, 1], inplace=True)

titanic_data=titanic_data.astype(float)
titanic_data.dtypes


# Use Survived as the target variable for the ANN Classification.
# Create a training and testing subset of the data
data1=titanic_data[['Class', 'Sex', 'Age', 'Survived']]
titanic_data_train, titanic_data_test, titanic_y_train, titanic_y_test = train_test_split(
    data1.iloc[:,1:3], data1.Survived, test_size=0.3) 

# Standardize the scaling of the variables by
# computing the mean and std to be used for later scaling.
scaler = preprocessing.StandardScaler()
scaler.fit(titanic_data_train)

# Perform the standardization process
titanic_data_train = scaler.transform(titanic_data_train)
titanic_data_test = scaler.transform(titanic_data_test)

# Assess your model using a confusion matrix and 
# a classification report. 
nnclass1 = MLPClassifier(activation='relu', solver='sgd', 
                         hidden_layer_sizes=(30, 30))
nnclass1.fit(titanic_data_train, titanic_y_train)
nnclass1_pred = nnclass1.predict(titanic_data_test)
nnclass1_pred

cm = metrics.confusion_matrix(titanic_y_test, nnclass1_pred)
print(cm)
plt.matshow(cm)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Value')
plt.ylabel('Actual Value')
plt.xticks([0, 1], ['Died','Survived'])
plt.yticks([0, 1], ['Died','Survived'])

print(metrics.classification_report(titanic_y_test, nnclass1_pred))


# Compare ANN Classification with the Classification Tree
## Create Classification Tree for titanic_data_train, 
# using min_samples_split=5 and min_samples_leaf=5

titanic_data['Class']=titanic_data['Class'].astype(np.int64)
titanic_data['Sex']=titanic_data['Sex'].astype(np.int64)
titanic_data['Age']=titanic_data['Age'].astype(np.int64)
titanic_data['Survived']=titanic_data['Survived'].astype(object)
titanic_data.Survived.replace([0, 1],['No', 'Yes'], inplace=True)
titanic_data.dtypes

col_names = list(titanic_data.columns.values)
col_names
classnames = list(titanic_data.Survived.unique())
classnames

tre1 = tree.DecisionTreeClassifier(min_samples_split=5,min_samples_leaf=5).fit(titanic_data.iloc[:,1:3],titanic_data['Survived'])

dot_data = StringIO()
tree.export_graphviz(tre1, out_file=dot_data,
                     feature_names=col_names[1:3],
                     class_names=classnames,
                     filled=True,
                     rounded=True,
                     special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
display(Image(graph.create_png()))

#Evaluation for tree
predicted = tre1.predict(titanic_data.iloc[:,1:3])
print(metrics.classification_report(titanic_data['Survived'], predicted))

cm1 = metrics.confusion_matrix(titanic_data['Survived'], predicted)
print(cm1)

plt.matshow(cm1)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Value')
plt.ylabel('Actual Value')
plt.xticks([0, 1], ['Died','Survived'])
plt.yticks([0, 1], ['Died','Survived'])


