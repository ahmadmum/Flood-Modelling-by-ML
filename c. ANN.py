# -*- coding: utf-8 -*-
"""
Created on %(1st July 2021 )s

@author: %(Mumtaz Ahmad)s
"""

#neural network

import pandas as pd #to handle the data
from keras.models import Sequential #tensorflow lib
from keras.layers import Dense #neural network 
from sklearn.preprocessing import StandardScaler


##### reading file
mydir = 'C:/Users/mumta/Desktop/SP/River Var Data/'
dataframe = pd.read_csv(mydir+"Vesubei_all.csv" ,header=0,parse_dates=[0])

dataframe = (dataframe.set_axis(['Time', 'd', 'p', 't', 'e'], axis=1))


#spliting train & test
train = dataframe[0:1096]
test  = dataframe[1096:1461]

#train and test date for plotting
train_dates = pd.to_datetime(train['Time'])
train_dates = train_dates.reset_index()
del train_dates['index']

test_dates = pd.to_datetime(test['Time'])
test_dates = test_dates.reset_index()
del test_dates['index']

#drop time column for scaling
train =  train.drop('Time', axis=1)
test =  test.drop('Time', axis=1)

# X & y 
X_train = train.drop('d', axis=1) # (1096 ,3)
y_train = train['d']    # (365 ,3)

X_test  = test.drop('d', axis=1) # (1096 ,1)
y_test  = test['d']  # (365 ,3)

#making dataframe



#normalizing
scaler=StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)




#################### define model
##### deeper and wider networks

# input layer
model = Sequential()
model.add(Dense(64, input_dim=3, activation='relu'))

# hidden layer
model.add(Dense(32, activation='relu'))  
model.add(Dense(16, activation='relu'))  
model.add(Dense(8, activation='relu'))  

#Output layer
model.add(Dense(1, activation='linear'))

#Compile ANN
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.summary()

history = model.fit(X_train_scaled, y_train, verbose=1 , epochs =50)


############################ Prediction

test_prediction = model.predict(X_test_scaled[:365])
train_prediction =  model.predict(X_train_scaled[:1096])





#from matplotlib import pyplot as plt
#from pandas import read_csv
#import math
#import seaborn as sns
#import numpy as np
#import matplotlib.pyplot as plt
#from math import sqrt
#from sklearn.preprocessing import MinMaxScaler
#from sklearn.metrics import mean_squared_error
#from sklearn.metrics import mean_absolute_error
#import sklearn.metrics as metrics
