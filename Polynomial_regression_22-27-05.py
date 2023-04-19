# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 23:01:45 2023

@author: virendra Singh Kaira
Data Sceince
Registration number: 22-27-05
"""
#Importing libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


# Reading the House Prediction data

df = pd.read_csv("E:\DS_ACADEMIC\DS2\ML\Data_Set\Housepriceprediction.csv")

# Normalizing the  House Prediction data

from sklearn import preprocessing
df_new=preprocessing.normalize(df.values,axis=0)
scaled_df = pd.DataFrame(df_new, columns=df.columns)
y=scaled_df.SalePrice
x=scaled_df.drop(['SalePrice','Id'],axis=1)


# Spliting the data into trainning and testing sets

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=4)

# Applying Polynomial Features of degree 2

from sklearn.preprocessing import PolynomialFeatures
model_poly = PolynomialFeatures(degree=2)
x_poly_train=model_poly.fit_transform(x_train)
x_poly_test=model_poly.fit_transform(x_test)

# Model Fitting and Prediction

model_poly_lin=LinearRegression()
model_poly_lin.fit(x_poly_train,y_train)
y_pred = model_poly_lin.predict(x_poly_test)

# Printong of r2 sccore 

poly_r2= r2_score(y_test,y_pred)*100
print("R2 from POLY learn:",poly_r2)

# Printing of mean squared error

mean_squared_error_poly_reg=mean_squared_error(y_test, y_pred)
print("Mean Squared Error for test set:",mean_squared_error_poly_reg)

# Visualization of Prediction and test data

t = np.linspace(1,438,438)
cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(8,6))
m1 = plt.scatter(t, y_test, color=cmap(0,9), s=10)
plt.plot(t,y_pred,color='blue',linewidth=0.5,label='Prediction')
plt.show()