# -*- coding: utf-8 -*-
"""
Created on Sat Nov  8 21:26:46 2014

@author: atproofer
"""


# -*- coding: utf-8 -*-
"""
Created on Sat Oct 25 10:16:30 2014

@author: atproofer

Scratch work for DoML BikeShare kaggle competition
"""

import csv
import numpy as np
from sklearn import ensemble
import matplotlib.pyplot as plt
from numpy import *

bike_share = genfromtxt('train.csv', delimiter=',',dtype=None)
bike_share_test = genfromtxt('test.csv', delimiter=',',dtype=None)
labels = bike_share[0,:]
print labels

train = bike_share[1:,:]
test = bike_share_test[1:,:]

date_time= train[:,0]
year= []
month= []
day=[]
hour=[]
for i in range(len(date_time)):
    year=year+ [date_time[i][0:4]]
    month=month+ [date_time[i][5:7]]
    day=day+ [date_time[i][8:10]]
    hour=hour+ [date_time[i][11:13]]
    
date_time_test= test[:,0]
year_test= []
month_test= []
day_test=[]
hour_test=[]
for i in range(len(date_time_test)):
    year_test  = year_test+ [date_time_test[i][0:4]]
    month_test = month_test+ [date_time_test[i][5:7]]
    day_test   = day_test+ [date_time_test[i][8:10]]
    hour_test  = hour_test+ [date_time_test[i][11:13]]

a = np.vstack((year,month,day,hour))
b=np.transpose(a)
b=float_(b)
Xtrain = float_(train[:,1:9])
Xtrain = np.hstack((b,Xtrain))

c = np.vstack((year_test,month_test,day_test,hour_test))
d = np.transpose(c)
d = float_(d)
Xtest = float_(test[:,1:9])
Xtest = np.hstack((d,Xtest))

casual = float_(train[:,9])
registered = float_(train[:,10])
count = float_(train[:,11])
y = count

Xw_train =Xtrain[((Xtrain[:,5]==0) & (Xtrain[:,6] == 1)), :]
Xnw_train =Xtrain[((Xtrain[:,5]!=0) | (Xtrain[:,6] != 1)), :]
Xw_test =Xtest[((Xtest[:,5]==0) & (Xtest[:,6] == 1)), :]
Xnw_test =Xtest[((Xtest[:,5]!=0) | (Xtest[:,6] != 1)), :]

yw_train =  y[((Xtrain[:,5]==0) & (Xtrain[:,6] == 1))]
ynw_train = y[((Xtrain[:,5]!=0) | (Xtrain[:,6] != 1))]

tree_w = ensemble.RandomForestRegressor(n_estimators = 50, max_depth = 15)
tree_nw = ensemble.RandomForestRegressor(n_estimators = 50, max_depth = 15)

tree_w.fit(Xw_train, yw_train)
tree_nw.fit(Xnw_train, ynw_train)

#hw_train = tree_w.predict(Xw_train)
hnw_train = tree_nw.predict(Xnw_train)
#plt.scatter(yw_train, hw_train)
#plt.scatter(ynw_train, hnw_train)

h = []
for i in range(len(Xtest)):
    if ((Xtest[i,5]==0) & (Xtest[i,6] == 1)):
        h.extend(np.maximum(tree_w.predict(Xtest[i,:]),0))
    else:
        h.extend(np.maximum(tree_nw.predict(Xtest[i,:]),0))


h = np.array(h)
h = int_(h)
date_time_test= date_time_test[:,np.newaxis]
h = h[:,np.newaxis]
test_output = np.hstack((date_time_test,h))
column_headers = np.array(['datetime', 'count'])
test_output = np.vstack((column_headers, test_output))
np.savetxt("foo.csv", test_output, delimiter=",",fmt="%s")
