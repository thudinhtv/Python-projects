#Read in library
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

#For the tree
from sklearn.feature_extraction.image import grid_to_graph
from sklearn import tree
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus
          
#Set working directory
os.chdir('Path\\Data')
os.getcwd()


###Data source (from THA01):
hospital = pd.read_table('CaliforniaHospitalData.csv', sep=',')
personnel = pd.read_table('CaliforniaHospitalData_Personnel.txt', sep='\t')
merged_data = pd.merge(hospital, personnel, on='HospitalID')
merged_data.drop(['Work_ID', 'PositionID', 'Website'], axis=1, inplace=True)     
merged_data.rename(columns={'NoFTE':'FullTimeCount', 'NetPatRev':'NetPatientRevenue', 
                                  'InOperExp':'InpatientOperExp', 'OutOperExp':'OutpatientOperExp', 
                                  'OperRev':'Operating_Revenue', 'OperInc':'Operating_Income'}, 
                   inplace=True)

newrows = [{'HospitalID':37436, 'Name':'Fallbrook Hospital', 'Zip':'92028', 'TypeControl':'District', 
            'Teaching':'Small/Rural', 'DonorType':'Charity', 'FullTimeCount':501, 
            'NetPatientRevenue':108960.418, 'InpatientOperExp':23001687.34, 'OutpatientOperExp':14727466.66, 
            'Operating_Revenue':40329849, 'Operating_Income':2600695,'AvlBeds':146, 'LastName':'Dinh', 'FirstName':'Thu', 
            'Gender':'F', 'PositionTitle':'State Board Representative', 'Compensation':89473, 'MaxTerm':3, 'StartDate':'1/24/2019'}, 
           
           {'HospitalID':28283, 'Name':'Hi-Desert Medical Center', 'Zip':'92252', 'TypeControl':'District', 
            'Teaching':'Small/Rural', 'DonorType':'Charity', 'FullTimeCount':451.5,'NetPatientRevenue':145733.5765, 
            'InpatientOperExp':31842679.16, 'OutpatientOperExp':21184931.84, 'Operating_Revenue':53619042, 
            'Operating_Income':591431, 'AvlBeds':179, 'LastName':'Dinh', 'FirstName':'Thu', 'Gender':'F', 
            'PositionTitle':'State Board Representative', 'Compensation':89473, 'MaxTerm':3, 'StartDate':'1/24/2019'}]

hospital_data = merged_data.append(newrows, ignore_index=True, sort=False)
hospital_data.shape


#Using numerical columns, conduct PCA and obtain the eigenvalues
##Define numerical columns:
hospital_data.dtypes
hospital_data.describe(include=['number'])
##PCA
hospital_reduct_pca = hospital_data[['HospitalID', 'FullTimeCount', 'NetPatientRevenue', 'InpatientOperExp',
       'OutpatientOperExp', 'Operating_Revenue', 'Operating_Income', 'AvlBeds', 'Compensation', 'MaxTerm']]
pca_result = pca(n_components=10).fit(hospital_reduct_pca)

#Obtain eigenvalues
pca_result.explained_variance_

#Components from the PCA
pca_result.components_

#Scree plot
plt.figure(figsize=(7,5))
plt.plot([1,2,3,4,5,6,7,8,9,10], pca_result.explained_variance_ratio_, '-o')
plt.ylabel('Proportion of Variance Explained') 
plt.xlabel('Principal Component') 
plt.xticks([1,2,3,4,5,6,7,8,9,10])

#Factor Analysis using Varimax rotation
hospital_reduct_fac = hospital_data[['HospitalID', 'FullTimeCount', 'NetPatientRevenue', 'InpatientOperExp',
       'OutpatientOperExp', 'Operating_Revenue', 'Operating_Income', 'AvlBeds', 'Compensation', 'MaxTerm']]

##Method1: using FactorAnalysis from sklearn
fact_result = fact(n_components=10).fit(hospital_reduct_fac)
fact_result.components_
print(pd.DataFrame(fact_result.components_, hospital_reduct_fac.columns))

##Method2: using FactorAnalyzer from factor_analyzer
from factor_analyzer import FactorAnalyzer
fa = FactorAnalyzer()
fa.analyze(hospital_reduct_fac, 10, rotation='varimax')
fa.loadings


#k-means clutter analysis for all numerical data
#Look at unique values of categorical variables
hospital_data.Teaching.unique()
hospital_data.DonorType.unique()
hospital_data.Gender.unique()
hospital_data.TypeControl.unique()
hospital_data.PositionTitle.unique()

#K-Means, 2 clusters
km = cls.KMeans(n_clusters=2).fit(hospital_data.loc[:,['FullTimeCount', 'NetPatientRevenue', 'InpatientOperExp',
       'OutpatientOperExp', 'Operating_Revenue', 'Operating_Income', 'AvlBeds', 'Compensation', 'MaxTerm']])
km.labels_

#Confusion matrix for Teaching (2-levels); DonorType (2-levels)
##Replace values
hospital_data.Teaching.replace(['Small/Rural', 'Teaching'],[0,1], inplace=True)
hospital_data.DonorType.replace(['Charity', 'Alumni'],[0,1], inplace=True)

##Convert back to categorical
hospital_data['Teaching'] = hospital_data['Teaching'].astype('category')
hospital_data['DonorType'] = hospital_data['DonorType'].astype('category')

##Confusion matrix for Teaching
cm_Teaching = metcs.confusion_matrix(hospital_data.Teaching, km.labels_)
print(cm_Teaching) 

plt.matshow(cm_Teaching)
plt.title('Confusion Matrix for Teaching')
plt.xlabel('Actual Value')
plt.ylabel('Predicted Value')

##Confusion matrix for DonorType
cm_DonorType = metcs.confusion_matrix(hospital_data.DonorType, km.labels_)
print(cm_DonorType)

plt.matshow(cm_DonorType)
plt.title('Confusion Matrix for DonorType')
plt.xlabel('Actual Value')
plt.ylabel('Predicted Value')


#K-Means, 4 clusters
km2 = cls.KMeans(n_clusters=4).fit(hospital_data.loc[:,['FullTimeCount', 'NetPatientRevenue', 'InpatientOperExp',
       'OutpatientOperExp', 'Operating_Revenue', 'Operating_Income', 'AvlBeds', 'Compensation', 'MaxTerm']])
km2.labels_

#Confusion matrix for TypeControl
hospital_data.TypeControl.replace(['District', 'Non Profit', 'Investor', 'City/County'],[2,1,3,0], inplace=True)
hospital_data['TypeControl'] = hospital_data['TypeControl'].astype('category')

cm_TypeControl = metcs.confusion_matrix(hospital_data.TypeControl, km2.labels_)
print(cm_TypeControl)       

plt.matshow(cm_TypeControl)
plt.title('Confusion Matrix for TypeControl')
plt.xlabel('Actual Value')
plt.ylabel('Predicted Value')



