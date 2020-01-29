#Read in library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.feature_extraction.image import grid_to_graph
from sklearn import tree
from sklearn import metrics

from sklearn.externals.six import StringIO
from IPython.display import Image, display
import pydotplus

#Set working directory
os.chdir('path\\Data')
os.getcwd()

#1. Regression Tree
#Read in data
reduc_data = pd.read_table('reduction_data_new.txt')
reduc_data.columns
reduc_data.dtypes
reduc_data.head()
reduc_data.isna().sum()

#Remove all the columns that have missing values
reduc_data1 = reduc_data[['peruse01', 'peruse02', 'peruse03', 'peruse04', 'peruse05',
                         'peruse06', 'pereou01', 'pereou02', 'pereou03', 'pereou04', 
                         'pereou05','pereou06', 'intent02', 'intent03', 'intent01']]

#Create Regression Tree using intent01 as the target variable, min_samples_split=20, min_samples_leaf=20
col_names = list(reduc_data1.iloc[:,0:14].columns.values)
col_names

tre = tree.DecisionTreeRegressor(min_samples_split=20,min_samples_leaf=20).fit(reduc_data1.iloc[:,0:14],reduc_data1.intent01)

dot_data = StringIO()
tree.export_graphviz(tre, out_file=dot_data,
                     feature_names=col_names,
                     filled=True,
                     rounded=True,
                     special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
display(Image(graph.create_png()))

#Create plot for your tree (run all together)
plt.scatter(reduc_data1.intent03, reduc_data1.intent01, color='DarkBlue')
plt.plot([4.5,4.5], [0,7], color='Green')
plt.plot([6.5,6.5], [0,7], color='Green')
plt.plot([0,4.5],[2.818,2.818], color='Red')
plt.plot([4.5, 6.5],[6.252, 6.252], color='Red')
plt.xlabel('intent03')
plt.ylabel('intent01')

#2. Classification trees
# Read in data
titanic_data = pd.read_csv('titanic_data.txt', sep='\t')
titanic_data.columns
titanic_data.dtypes

# Convert the categorical data to numeric.                 
## Copy each column into a new object
num_cols = pd.DataFrame(titanic_data[['Class','Sex','Age']])

##Rename the columns to keep them distinct from original dataframe
num_cols.rename(columns={'Class':'Class2','Sex':'Sex2','Age':'Age2'}, inplace=True)

##Obtain the values to be converted
num_cols['Class2'].unique()
num_cols['Sex2'].unique()
num_cols['Age2'].unique()

##Convert the values into numerical
num_cols['Class2'].replace(['1st','2nd','3rd','Crew'],[0,1,2,3], inplace=True)
num_cols['Sex2'].replace(['Female','Male'],[0,1], inplace=True)
num_cols['Age2'].replace(['Child','Adult'],[0,1], inplace=True)

titanic_data2 = pd.concat([titanic_data, num_cols], axis=1)
titanic_data2.columns
titanic_data2.dtypes
titanic_data2.head()

#Create Classification Tree for titanic_data2, using min_samples_split=5 and min_samples_leaf=5
col_names = list(titanic_data2.columns.values)
col_names
classnames = list(titanic_data2.Survived.unique())
classnames

tre1 = tree.DecisionTreeClassifier(min_samples_split=5,min_samples_leaf=5).fit(titanic_data2.iloc[:,4:7],titanic_data2['Survived'])

dot_data = StringIO()
tree.export_graphviz(tre1, out_file=dot_data,
                     feature_names=col_names[4:7],
                     class_names=classnames,
                     filled=True,
                     rounded=True,
                     special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
display(Image(graph.create_png()))

# Provide Plot for the tree (run all together)
plt.scatter(titanic_data2.Sex2, titanic_data2.Age2, color='DarkBlue')
plt.plot([0.5,0.5], [0, 1], color='Green')
plt.plot([0,1], [0.5, 0.5], color='Red')
plt.xlabel('Sex')
plt.ylabel('Age')


#Prune the tree: Create Classification Tree for titanic_data2 
##using min_samples_split=50 and min_samples_leaf=50
col_names = list(titanic_data2.columns.values)
classnames = list(titanic_data2.Survived.unique())

tre2 = tree.DecisionTreeClassifier(min_samples_split=50,min_samples_leaf=50).fit(titanic_data2.ix[:,4:7],titanic_data2['Survived'])

dot_data = StringIO()
tree.export_graphviz(tre2, out_file=dot_data,
                     feature_names=col_names[4:7],
                     class_names=classnames,
                     filled=True,
                     rounded=True,
                     special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
display(Image(graph.create_png()))

##Evaluation for tre2
predicted = tre2.predict(titanic_data2.ix[:,4:7])
print(metrics.classification_report(titanic_data2['Survived'], predicted))

cm = metrics.confusion_matrix(titanic_data2['Survived'], predicted)
print(cm)

plt.matshow(cm)
plt.title('Confusion Matrix')
plt.xlabel('Actual Value')
plt.ylabel('Predicted Value')
plt.xticks([0,1], ['Yes','No'])

## using min_samples_split=100 and min_samples_leaf=100
col_names = list(titanic_data2.columns.values)
classnames = list(titanic_data2.Survived.unique())

tre3 = tree.DecisionTreeClassifier(min_samples_split=100,min_samples_leaf=100).fit(titanic_data2.ix[:,4:7],titanic_data2['Survived'])

dot_data = StringIO()
tree.export_graphviz(tre3, out_file=dot_data,
                     feature_names=col_names[4:7],
                     class_names=classnames,
                     filled=True,
                     rounded=True,
                     special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
display(Image(graph.create_png()))

#Evaluation for tre3
predicted = tre3.predict(titanic_data2.ix[:,4:7])
print(metrics.classification_report(titanic_data2['Survived'], predicted))

cm = metrics.confusion_matrix(titanic_data2['Survived'], predicted)
print(cm)

plt.matshow(cm)
plt.title('Confusion Matrix')
plt.xlabel('Actual Value')
plt.ylabel('Predicted Value')
plt.xticks([0,1], ['Yes','No'])