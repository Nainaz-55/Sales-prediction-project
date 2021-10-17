#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 18:31:35 2021

@author: nainaz
"""



#importing the librery
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



#reading the data from your files
data=pd.read_csv('advertising.csv')
data.head() #To see the top 5 rows of data set
#identify what is input and what is output :: sales is dependent on how much the invested in TV Radion Newspaper

#visualizing the data set
#visualize we need X-axis and Y-axis
#Here we need sub plots (1-row,3-col,sharing Y-axis bcz on Y we gonna put sales bcz sales is common,axs[0] for first plot 1st row)
fig, axs = plt.subplots(1,3,sharey =True)#here we creat 1 row 3col space
data.plot(kind='scatter', x='TV', y='Sales',ax=axs[0], figsize=(14,7))#14 is width and 7 is height of figure
data.plot(kind='scatter', x='Radio', y='Sales',ax=axs[1])#figsize is universal to all no need to give again
data.plot(kind='scatter', x='Newspaper', y='Sales',ax=axs[2])

#give this to the ML algm . bfr tht do the transformation of data
#Leaner regression algm tekes 2 inputs X-independent var and Y-dependent var
#CREATING X and Y for lenear regresssion
feature_cols= ['TV']
x = data[feature_cols]
y = data.Sales


#after transforming data we give it to ML algm
#IMPORTING Lenear regression ALGM FOR SIMPLE LINEAR REGRESSION
from sklearn.linear_model import LinearRegression
lr = LinearRegression()#giving alias name to LR module

#sending an X and Y created to algm using fit() function
lr.fit(x,y)

print(lr.intercept_)
print(lr.coef_)


#y=a+bx,, res=intercept + coef * x
#if I invest 50k$ then what will my sales(result)
result=6.9748214882298925 + 0.05546477*(50.0)


#creating a dataframe with min and max value of table
x_new=pd.DataFrame({'TV':[data.TV.min(),data.TV.max()]})
x_new.head()


#MULTIPLE LINEAR REGRESSION(EX:REDIO,TV, AND ALL)

preds = lr.predict(x_new)#predict() is used to predict the the values wt v provide
preds


#least squard line:::if data show linear relnp between x and y var,,,form vertical distance betweeen dataline
data.plot(kind='scatter',x='TV',y='Sales')
plt.plot(x_new,preds,c='red',linewidth=3)






#summary of the model btween sales and TV
import statsmodels.formula.api as smf
lm = smf.ols(formula = 'Sales ~ TV', data=data).fit()
lm.conf_int()


#FINDING THE PROBABILY VALUES
lm.pvalues
#FINDING THE R-SQUARED VALUES
lm.rsquared


#MULTI LINEAR REGRESSION
feature_cols = ['TV','Radio','Newspaper']
x=data[feature_cols]
y=data.Sales

lr= LinearRegression()
lr.fit(x,y)

print(lr.intercept_)
print(lr.coef_)


lm = smf.ols(formula = 'Sales ~ TV+Radio+Newspaper',data=data).fit()
lm.conf_int()
lm.summary()













