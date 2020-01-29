#Read in Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

#For QQ Plot
import scipy.stats as sts

#Correlation p-values
from scipy.stats.stats import pearsonr

#Regression output
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as smf

# Setup the Working Directory
os.chdir('Path\\Data')
os.getcwd()

# Read in data
reduc_data = pd.read_table('reduction_data_new.txt')
reduc_data.columns
reduc_data.dtypes

# 1. Using intent01 as the target variable, create a regression model 
# using only 5 other numerical variables
data=reduc_data[['peruse02', 'peruse04', 'peruse05', 'pereou03', 'pereou04', 'intent01']]
data.dtypes
linreg = smf.ols('intent01 ~ peruse02 + peruse04 + peruse05 + pereou03 + pereou04', data).fit()
linreg.summary()

#2. Assess the validity of assumptions
## Assumption1: Linearity
data.plot.scatter(x='peruse02', y='intent01')
data.plot.scatter(x='peruse04', y='intent01')
data.plot.scatter(x='peruse05', y='intent01')
data.plot.scatter(x='pereou03', y='intent01')
data.plot.scatter(x='pereou04', y='intent01')

## Assumption2: Multicollinearity
###1st method: Correlation analysis
data.corr()
pearsonr(data.peruse02, data.intent01)
pearsonr(data.peruse04, data.intent01)
pearsonr(data.peruse05, data.intent01)
pearsonr(data.pereou03, data.intent01)
pearsonr(data.pereou04, data.intent01)

###2nd method: VIF
linreg1 = LinearRegression(fit_intercept=True, normalize=True)

linreg1.fit(data[['peruse04', 'peruse05', 'pereou03', 'pereou04']], data.peruse02)
VIF1 = 1/(1-linreg1.score(data[['peruse04', 'peruse05', 'pereou03', 'pereou04']], data.peruse02))

linreg1.fit(data[['peruse02', 'peruse05', 'pereou03', 'pereou04']], data.peruse04)
VIF2 = 1/(1-linreg1.score(data[['peruse02', 'peruse05', 'pereou03', 'pereou04']], data.peruse04))

linreg1.fit(data[['peruse02', 'peruse04', 'pereou03', 'pereou04']], data.peruse05)
VIF3 = 1/(1-linreg1.score(data[['peruse02', 'peruse04', 'pereou03', 'pereou04']], data.peruse05))

linreg1.fit(data[['peruse02', 'peruse04', 'peruse05', 'pereou04']], data.pereou03)
VIF4 = 1/(1-linreg1.score(data[['peruse02', 'peruse04', 'peruse05', 'pereou04']], data.pereou03))

linreg1.fit(data[['peruse02', 'peruse04', 'peruse05', 'pereou03']], data.pereou04)
VIF5 = 1/(1-linreg1.score(data[['peruse02', 'peruse04', 'peruse05', 'pereou03']], data.pereou04))

print('VIF peruse02: ', VIF1,
       '\nVIF peruse04: ', VIF2,
       '\nVIF peruse05: ', VIF3,
       '\nVIF pereou03: ', VIF4,
       '\nVIF pereou04: ', VIF5)

##Assumption3: Homoscedasticity (constant variance)
plt.scatter(linreg.fittedvalues, linreg.resid)
plt.xlabel('Predicted/Fitted Values')
plt.ylabel('Residual Values')
plt.title('Assessing Homoscedasticity')
plt.show()

##Assumption4: Independence -> see Durbin Watson result from regression output table

##Assumption5: QQ plot for Normality
sts.probplot(linreg.resid, dist="norm", plot=plt)

#2. Model's F-test and T-tests: see result from regression output

#3. Model Equation: use coefficient from regression output


