
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 25 10:16:30 2014

@author: atproofer

Scratch work for DoML BikeShare kaggle competition
"""

import csv
import numpy as np
from sklearn import linear_model, ensemble
import matplotlib.pyplot as plt
from numpy import *


bike_share = genfromtxt('train.csv', delimiter=',',dtype=None)

labels = bike_share[0,:]
print labels

data = bike_share[1:,:]
data=np.random.permutation(data)

date_time= data[:,0]
year= []
month= []
day=[]
hour=[]
for i in range(len(date_time)):
    year=year+ [date_time[i][0:4]]
    month=month+ [date_time[i][5:7]]
    day=day+ [date_time[i][8:10]]
    hour=hour+ [date_time[i][11:13]]

a = np.vstack((year,month,day,hour))
b=np.transpose(a)
#print b[:5,:]
b=float_(b)
#print data[:5,:9]
X = float_(data[:,1:9])
X=np.hstack((b,X))
#print X[:2,:]
#print X[:5,:]
#input_x = [0,1,2,3,4,5,6,7]

#X = X[:,input_x]

casual = float_(data[:,9])
registered = float_(data[:,10])
count = float_(data[:,11])
y = count
#print X[:5,:]
ntrain = 6000;

Xtrain = X[:ntrain]
Xtest = X[ntrain:]
ytrain = y[:ntrain] 
ytest = y[ntrain:]

# Create linear regression object
regr = linear_model.LinearRegression()
ridge_regr = linear_model.Ridge(alpha=10)
#b_ridge = linear_model.BayesianRidge()
#lasso_r = linear_model.Lasso()
#lasr_r = linear_model.LARS()
tree  = ensemble.RandomForestRegressor(n_estimators = 25, max_depth = 15)
tree2 = ensemble.RandomForestRegressor(n_estimators = 50, max_depth = 15)
tree3 = ensemble.RandomForestRegressor(n_estimators = 100, max_depth = 15)
tree4 = ensemble.RandomForestRegressor(n_estimators = 200, max_depth = 15)


# Train the model using the training sets
regr.fit(Xtrain, ytrain)
tree.fit(Xtrain, ytrain)
tree2.fit(Xtrain, ytrain)
tree3.fit(Xtrain, ytrain)
tree4.fit(Xtrain, ytrain)
#ridge_regr.fit(Xtrain,ytrain)
#b_ridge.fit(Xtrain,ytrain)
#lasso_r.fit(Xtrain,ytrain)
#lasr_r.fit(Xtrain,ytrain)
# The coefficients
#print("Coefficients:", regr.coef_)
#print("Constant Term: %f" %regr.intercept_ )
# The mean square error
#print("Residual sum of squares regr: %.2f"
#      % np.mean((regr.predict(Xtest) - ytest) ** 2))
def rmsle(h, y):
    return np.sqrt(np.square(np.log(h + 1) - np.log(y + 1)).mean())
    
print("Root Mean Square Log Error tree: %.2f"
      % rmsle(tree.predict(Xtest),ytest))
print("Root Mean Square Log Error tree2: %.2f"
      % rmsle(tree2.predict(Xtest),ytest))
print("Root Mean Square Log Error tree3: %.2f"
      % rmsle(tree3.predict(Xtest),ytest))
print("Root Mean Square Log Error tree4: %.2f"
      % rmsle(tree4.predict(Xtest),ytest))
#print("Residual sum of squares ridge: %.2f"
#      % np.mean((ridge_regr.predict(Xtest) - ytest) ** 2))
#print("Residual sum of squares b_ridge: %.2f"
#      % np.mean((b_ridge.predict(Xtest) - ytest) ** 2))
#print("Residual sum of squares lasso: %.2f"
#      % np.mean((lasso_r.predict(Xtest) - ytest) ** 2))
#print("Residual sum of squares lars: %.2f"
#      % np.mean((lasr_r.predict(Xtest) - ytest) ** 2))

# Explained variance score: 1 is perfect prediction
#print('Variance score: %.2f' % regr.score(Xtest, ytest))
#plt.plot(X[:,0],y)
#plt.show
"""
# Plot outputs
plt.scatter(Xtest, ytest,  color='black')
plt.scatter(Xtrain,ytrain, color='red')
plt.plot(Xtest, tree.predict(Xtest), color='blue', linewidth=3)
#regr.fit(Xtest, ytest)
#plt.plot(Xtest,regr.predict(Xtest), color= 'orange', linewidth=3)
#plt.plot(Xtrain, regr.predict(Xtrain), color='orange', linewidth=3)
plt.title("Bike Usage against condition")
plt.ylabel("Count")
plt.xlabel(labels[input_x+1])
plt.show()"""