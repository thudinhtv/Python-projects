#Read in library

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

#Binning of data
from scipy.stats import binned_statistic

#Regression output
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as smf

#Set working directory
os.chdir('Path\\Data')
os.getcwd()

# Read in data
hospital = pd.read_table('CaliforniaHospitalData.csv', sep=',')
personnel = pd.read_table('CaliforniaHospitalData_Personnel.txt', sep='\t')
merged_data = pd.merge(hospital, personnel, on='HospitalID')
merged_data.shape
merged_data.dtypes

# Convert Compensation column to categorical variable
merged_data['Compensation'] = merged_data['Compensation'].astype('category')


# Subset columns and save as a new dataframe
data=merged_data[['Teaching', 'DonorType', 'TypeControl','Gender', 'PositionTitle', 'Compensation','AvlBeds','OperInc', 'OperRev']]
data.shape
data.dtypes
data.isna().sum()

# Perform binning for Avlbeds
data['AvlBeds'].max()
data['AvlBeds'].min()
data['AvlBeds'].plot.hist(alpha=0.5)

bin_counts,bin_edges,binnum = binned_statistic(data['AvlBeds'], 
                                               data['AvlBeds'], 
                                               statistic='count', 
                                               bins=32)

bin_counts
bin_edges

bin_interval = [12, 40, 70, 130, 440, 910]
bin_counts,bin_edges,binnum = binned_statistic(data['AvlBeds'], 
                                               data['AvlBeds'], 
                                               statistic='count', 
                                               bins=bin_interval)

bin_counts
bin_edges


# Recode the values in the age column based on the binning
binlabels = ['AvlBeds_12_39', 'AvlBeds_40_69', 'AvlBeds_70_129', 'AvlBeds_130_439', 'AvlBeds_440_909']
AvlBeds_catego = pd.cut(data['AvlBeds'], bin_interval, right=False, retbins=False, labels=binlabels)
AvlBeds_catego.name = 'AvlBeds_catego'

# Take the binning data and add it as a column to the dataframe
data = data.join(pd.DataFrame(AvlBeds_catego))

# Compare the original AvlBeds column to the AvlBeds_catego
data[['AvlBeds', 'AvlBeds_catego']].sort_values(by='AvlBeds')
data.dtypes

# Create dummy variables for the column AvlBeds_catego
AvlBeds_dummy = pd.get_dummies(data['AvlBeds_catego'])
AvlBeds_dummy.head()
data = data.join(AvlBeds_dummy)

# Create dummy variables for the column Compensation
data['Compensation'].unique()
Compensation_dummy = pd.get_dummies(data['Compensation'], prefix='Comp')
Compensation_dummy.head()
data = data.join(Compensation_dummy)


# Create dummy variables for the column Teaching
data['Teaching'].unique()
Teaching_dummy = pd.get_dummies(data['Teaching'], prefix='Teach')
Teaching_dummy.head()
Teaching_dummy.columns = ['Teach_Small_Rural','Teach_Teaching']
data = data.join(Teaching_dummy)

# Create dummy variables for the column DonorType
data['DonorType'].unique()
DonorType_dummy = pd.get_dummies(data['DonorType'], prefix='Donor')
DonorType_dummy.head()
data = data.join(DonorType_dummy)

# Create dummy variables for the column TypeControl
data['TypeControl'].unique()
TypeControl_dummy = pd.get_dummies(data['TypeControl'], prefix='TypeCon')
TypeControl_dummy.head()
TypeControl_dummy.columns = ['TypeCon_City_County','TypeCon_District', 'TypeCon_Investor', 'TypeCon_Non Profit' ]
data = data.join(TypeControl_dummy)

# Create dummy variables for the column Gender
data['Gender'].unique()
Gender_dummy = pd.get_dummies(data['Gender'], prefix='Gender')
Gender_dummy.head()
data = data.join(Gender_dummy)

# Create dummy variables for the column PositionTitle
data['PositionTitle'].unique()
PositionTitle_dummy = pd.get_dummies(data['PositionTitle'], prefix='Position')
PositionTitle_dummy.head()
PositionTitle_dummy.columns = ['Position_ActDirector','Position_RegionalRep','Position_SafetyIns','Position_StateRep']
data = data.join(PositionTitle_dummy)

data.columns

# Model 1: Regression Model using OperInc as dependent variable 
# and all dummies as independent variables
data_reg1 = smf.ols('OperInc ~ Teach_Small_Rural + Donor_Alumni + TypeCon_City_County + TypeCon_District + TypeCon_Investor + Gender_F + Position_ActDirector + Position_RegionalRep + Position_SafetyIns + Comp_23987 + Comp_46978 + Comp_89473 + AvlBeds_12_39 + AvlBeds_40_69 + AvlBeds_70_129 + AvlBeds_130_439', data).fit()
data_reg1.summary()

# Model 2: Remove the insignificant variables from Model 1
# (Teaching dummies, Gender dummies, Position_Title dummies, AvlBeds dummies) 
# and rerun the regression
data_reg2 = smf.ols('OperInc ~ Donor_Alumni + TypeCon_City_County + TypeCon_District + TypeCon_Investor + Comp_23987 + Comp_46978 + Comp_89473', data).fit()
data_reg2.summary()


# Model 3: Regression Model using OperRev as dependent variable
# and all dummies as independent variables
data_reg3 = smf.ols('OperRev ~ Teach_Small_Rural + Donor_Alumni + TypeCon_City_County + TypeCon_District + TypeCon_Investor + Gender_F + Position_ActDirector + Position_RegionalRep + Position_SafetyIns + Comp_23987 + Comp_46978 + Comp_89473 + AvlBeds_12_39 + AvlBeds_40_69 + AvlBeds_70_129 + AvlBeds_130_439', data).fit()
data_reg3.summary()

# Model 4: Remove the insignificant variables from Model 3
# (Teaching dummies, Gender dummies) 
# and rerun the regression
data_reg4 = smf.ols('OperRev ~ Donor_Alumni + TypeCon_City_County + TypeCon_District + TypeCon_Investor + Position_ActDirector + Position_RegionalRep + Position_SafetyIns + Comp_23987 + Comp_46978 + Comp_89473 + AvlBeds_12_39 + AvlBeds_40_69 + AvlBeds_70_129 + AvlBeds_130_439', data).fit()
data_reg4.summary()
