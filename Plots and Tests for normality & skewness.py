import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.stats as sts             

os.chdir('C:\\working path\\Data')
os.getcwd()

pollute = pd.read_table('pollute.txt', sep='\t')
ozone_data = pd.read_table('ozone.data.txt')

#1.Summarizing the Data with Python
pollute.dtypes                  
pollute.describe()              
pollute.median()                
pollute.var()                   
pollute.isna().sum()            
pollute.nunique()               

#2.Using Plots with Python
#Boxplot for all variables:

pollute.boxplot()
plt.show()

#Scatterplot for all variables using 'Pollution' as the target variable:
##Population
pollute.plot.scatter(x='Population', y='Pollution', color='DarkBlue')      
plt.show()

##Industry
pollute.plot.scatter(x='Industry', y='Pollution', color='DarkGreen')      
plt.show()

##Wet.days
pollute.plot.scatter(x='Wet.days', y='Pollution', color='DarkGreen')      
plt.show()

###Temp
pollute.plot.scatter(x='Temp', y='Pollution', color='DarkBlue')      
plt.show()

###Wind
pollute.plot.scatter(x='Wind', y='Pollution', color='DarkBlue')      
plt.show()

###Rain
pollute.plot.scatter(x='Rain', y='Pollution', color='DarkBlue')      
plt.show()


#3.Assessing Normality with Python
sts.probplot(ozone_data.rad, dist="norm", plot=plt)    
plt.show()
sts.shapiro(ozone_data.rad)

sts.probplot(ozone_data.temp, dist="norm", plot=plt)   
plt.show()
sts.shapiro(ozone_data.temp)

sts.probplot(ozone_data.wind, dist="norm", plot=plt)   
plt.show()
sts.shapiro(ozone_data.wind)

sts.probplot(ozone_data.ozone, dist="norm", plot=plt)   
plt.show()
sts.shapiro(ozone_data.ozone)

#4.Skewness and Kurtosis with Python

##Pollution
plt.figure();
pollute['Pollution'].plot.hist(alpha=0.5)          
plt.show()

pollute.Pollution.skew()
pollute.Pollution.kurt()


##Temp
plt.figure();
pollute['Temp'].plot.hist(alpha=0.5)          
plt.show()

pollute.Temp.skew()
pollute.Temp.kurt()

##Industry
plt.figure();
pollute['Industry'].plot.hist(alpha=0.5)          
plt.show()

pollute.Industry.skew()
pollute.Industry.kurt()

##Population
plt.figure();
pollute['Population'].plot.hist(alpha=0.5)          
plt.show()

pollute.Population.skew()
pollute.Population.kurt()

##Wind
plt.figure();
pollute['Wind'].plot.hist(alpha=0.5)          
plt.show()

pollute.Wind.skew()
pollute.Wind.kurt()

#Rain
plt.figure();
pollute['Rain'].plot.hist(alpha=0.5)          
plt.show()

pollute.Rain.skew()
pollute.Rain.kurt()

#Wet.days
plt.figure();
pollute['Wet.days'].plot.hist(alpha=0.5)          
plt.show()

#Rename the column 'Wet.days' to 'Wet_days' since 'pollute.Wet.days.skew()'
##will not work because of the period (.) in the column name
pollute.rename(columns={'Wet.days':'Wet_days'}, inplace=True)     
pollute.columns

#Calulate the skewness and kurtosis
pollute.Wet_days.skew()
pollute.Wet_days.kurt()

#Rename the column back to 'Wet.days'
pollute.rename(columns={'Wet_days':'Wet.days'}, inplace=True)   
pollute.columns


