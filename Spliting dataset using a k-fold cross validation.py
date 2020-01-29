import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import KFold 

os.chdir('Path\\Data')
os.getcwd()

#1.Import the data and combine both into a single dataframe

hospital_data = pd.read_table('CaliforniaHospitalData.csv', sep=',')
personnel_data = pd.read_table('CaliforniaHospitalData_Personnel.txt', sep='\t')
hospital_data.shape
personnel_data.shape

merged_data = pd.merge(hospital_data, personnel_data, on='HospitalID')
merged_data.shape
merged_data.head(5)

#2.After the data has been merged, remove the following columns of data: duplicate columns, Work_ID, PositionID, Website 

merged_data.columns
merged_data.drop(['Work_ID', 'PositionID', 'Website'], axis=1, inplace=True)     
merged_data.columns

#3.Select only those hospitals that are “Small/Rural” and have 15 or more available beds. Exclude hospitals with a negative operating income. 
##Export your data as tab-delimited and name the file hospital_data_new.txt.

merged_data_new = merged_data[(merged_data.Teaching == "Small/Rural") & 
                              (merged_data.AvlBeds >= 15) & 
                              (merged_data.OperInc >= 0)]
merged_data_new.shape
merged_data_new.loc[:, ['Teaching', 'AvlBeds', 'OperInc']]
merged_data_new.to_csv('hospital_data_new.txt', index=None, sep='\t')

#4.Open the newly created file in Python as a new dataframe. Change the name of the following columns

hospital_data_df1 = pd.read_table('hospital_data_new.txt', sep ='\t')
hospital_data_df1.columns
hospital_data_df1.rename(columns={'NoFTE':'FullTimeCount', 'NetPatRev':'NetPatientRevenue', 
                                  'InOperExp':'InpatientOperExp', 'OutOperExp':'OutpatientOperExp', 
                                  'OperRev':'Operating_Revenue', 'OperInc':'Operating_Income'}, 
                         inplace=True)
hospital_data_df1.columns
hospital_data_df1.shape

#5.Select two of the existing hospitals in the data and create a new position for each hospital. 
##Insert yourself as the new employee at those two hospitals; put in your first name and last name. Put today’s date as the start date. 
###Select one of the positions as shown in the table below and fill out the data accordingly. 
####Fill in the rest of the columns as you choose. You should have two new rows of data.

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

hospital_data_df2 = hospital_data_df1.append(newrows, ignore_index=True, sort=False)
len(hospital_data_df1.index.values)
len(hospital_data_df2.index.values)
hospital_data_df2.loc[28:29, ['LastName', 'PositionTitle', 'Compensation', 'StartDate']]

#6.Convert any date-time columns into a datetime datatype. Confirm your changes by outputting the datatypes for all columns to your Python console.

hospital_data_df2.dtypes
hospital_data_df2['StartDate'] = pd.to_datetime(hospital_data_df2['StartDate'])
hospital_data_df2.dtypes

#7.Select all the Regional Representatives who work at a hospital with operating income greater than $100,000. 
##Save this as a new dataframe and then export it as a new file.

hospital_data_df3 = hospital_data_df2[(hospital_data_df2.PositionTitle =='Regional Representative') & 
                                      (hospital_data_df2.Operating_Income > 100000)]
hospital_data_df3.loc[:,['PositionTitle', 'Operating_Income']]
hospital_data_df3.to_csv('hospital_data_new2.txt', index=None, sep='\t')

#8.Using the original data, select all hospitals that are non-profit with more than 250 employees, unless the net patient revenue is smaller than $109,000. 
#Remove the columns containing employee information and save it as a new dataframe. Export the data as a new tab-delimited file

merged_data.shape
merged_data = merged_data[((merged_data.TypeControl=='Non Profit') & (merged_data.NoFTE > 250)) 
                          | (merged_data.NetPatRev < 109000)]
merged_data.shape
merged_data.loc[:, ['HospitalID', 'TypeControl', 'NoFTE', 'NetPatRev']]

merged_data.columns
hospital_data_df4 = merged_data.drop(['NoFTE', 'LastName', 'FirstName', 'Gender', 'PositionTitle', 
                                      'Compensation', 'MaxTerm', 'StartDate'], axis=1, inplace=False)  
hospital_data_df4.columns
hospital_data_df4.to_csv('hospital_data_new3.txt', index=None, sep='\t')

#9.Create a training and testing dataset by using a k-fold cross validation technique. Include up to 4 folds. 
#Export the training data and save it as training_data. 
#Export the test data as testing_data. Save both as comma-delimited files. 

kf = KFold(n_splits = 4)
hospital_data_df5 = hospital_data_df4.reset_index(drop = True)
hospital_data_df5.index
i = 0
for train, test in kf.split(hospital_data_df5.index):
    print("%s %s" % (train, test))
    i = i + 1
    training_data = hospital_data_df5.loc[train]
    training_data.to_csv('training_data_4fold_'+ str(i) + '.csv', index=None, sep=',')
    testing_data = hospital_data_df5.loc[test]
    testing_data.to_csv('testing_data_4fold_'+ str(i) + '.csv', index=None, sep=',')
  
