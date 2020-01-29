#2.Cluster Analysis with Python

##Set working directory
os.chdir('Path\\Data')
os.getcwd()

##Read in library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.decomposition import PCA as pca
from sklearn.decomposition import FactorAnalysis as fact

#Clustering modules
import sklearn.metrics as metcs
from scipy.cluster import hierarchy as hier
from sklearn import cluster as cls

##Read in data
car_data = pd.read_table("car.test.frame.txt", sep="\t")
car_data.columns
car_data.dtypes
car_data.head(5)

#Replace values of Type
car_data.Type.replace(['Small', 'Sporty', 'Compact', 'Medium', 'Large', 'Van'],[0,1,2,3,4,5], inplace=True)

#Convert back to categorical
car_data['Type'] = car_data['Type'].astype('category')
car_data.dtypes
car_data.Type.unique()

#K-Means Analysis
##k=6
km = cls.KMeans(n_clusters=6).fit(car_data.loc[:, ['Weight', 'Price', 'Disp.', 'HP']])
km.labels_

##k=4
km2 = cls.KMeans(n_clusters=4).fit(car_data.loc[:, ['Weight', 'Price','Disp.', 'HP']])
km2.labels_

##Assess the misclassification using a confusion matrix
cm = metcs.confusion_matrix(car_data.Type, km.labels_)
print(cm)

#Color-based chart
plt.matshow(cm)
plt.title('Confusion Matrix')
plt.xlabel('Actual Value')
plt.ylabel('Predicted Value')
plt.xticks([0,1,2,3,4,5], ['Small', 'Sporty', 'Compact', 'Medium', 'Large', 'Van'])
plt.show()

#Agglomerative clustering
agg1 = cls.AgglomerativeClustering(linkage='ward').fit(car_data[['Weight', 'Price','Disp.', 'HP']])
agg1.labels_

#Create a plot to view the output
z = hier.linkage(car_data[['Weight', 'Price','Disp.', 'HP']], 'single')
plt.figure()
dn = hier.dendrogram(z)
plt.show()
