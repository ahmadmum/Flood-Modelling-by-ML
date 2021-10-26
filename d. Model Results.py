# -*- coding: utf-8 -*-
"""
Created on %(1st July 2021 )s

@author: %(Mumtaz Ahmad)s
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt


mydir = 'C:/Users/mumta/Desktop/SP/River Var Data/'


# running ANN Reg best file
runfile('C:/Users/mumta/Desktop/SP/Python/3. ANN Reg best.py', wdir='C:/Users/mumta/Desktop/SP/Python')

#reading training file of NAM
train_nam = pd.read_csv(mydir+"Mike11.csv" ,header=0,parse_dates=[0])

#reading training file of NAM
predict_nam = pd.read_csv(mydir+"Mike11 predict.csv" ,header=0,parse_dates=[0])



#############################################      Plotting

################## ANN model

original = dataframe[['Time', 'd']]
original['Time']=pd.to_datetime(original['Time'])

# test

test_prediction_df = pd.DataFrame(test_prediction, columns = ['d'])
test_prediction_d = pd.concat([test_dates,test_prediction_df], axis=1)

original_test = original.loc[original['Time'] >= '2014-01-01']


fig = plt.figure(figsize=(6,5))
ax = fig.add_subplot(111)
ax.plot(original_test['Time'], original_test['d'], 'g', label='Observed', linestyle = 'solid', linewidth = 2)
ax.plot(test_prediction_d['Time'], test_prediction_d['d'], 'r', label='Prediction', linestyle = 'solid', linewidth = 2)

ax.legend()
ax.set_xlabel('Time')
ax.set_ylabel('Value (m3/s)')



# train


train_prediction_df = pd.DataFrame(train_prediction, columns = ['d'])
train_prediction_d = pd.concat([train_dates,train_prediction_df], axis=1)



original_train = original.loc[original['Time'] <= '2013-12-31']


fig = plt.figure(figsize=(6,5))
ax = fig.add_subplot(111)
ax.plot(original_train['Time'], original_train['d'], 'g', label='Observed', linestyle = 'solid', linewidth = 2)
ax.plot(train_prediction_d['Time'], train_prediction_d['d'], 'r', label='Prediction', linestyle = 'solid', linewidth = 2)

ax.legend()
ax.set_xlabel('Time')
ax.set_ylabel('Value (m3/s)')


# test, train


original_train_test = original.loc[original['Time'] >= '2011-01-01']


fig = plt.figure(figsize=(10,4))
ax = fig.add_subplot(111)
ax.plot(original_train_test['Time'], original_train_test['d'], 'g', label='Observed', linestyle = 'solid', linewidth = 2 )
ax.plot(train_prediction_d['Time'], train_prediction_d['d'], 'r', label='Calibration', linestyle = 'solid', linewidth = 2)
ax.plot(test_prediction_d['Time'], test_prediction_d['d'], 'y', label='Validation', linestyle = 'solid', linewidth = 2)

ax.legend()
ax.set_xlabel('Time')
ax.set_ylabel('Value (m3/s)')



############################ NAM

# test
fig = plt.figure(figsize=(6,5))
ax = fig.add_subplot(111)
ax.plot(original_test['Time'], original_test['d'], 'g', label='Observed', linestyle = 'solid', linewidth = 2)
ax.plot(predict_nam['Time'], predict_nam['Prediction'], 'r', label='Prediction', linestyle = 'solid', linewidth = 2)

ax.legend()
ax.set_xlabel('Time')
#ax.set_ylabel('Value (m3/s)')


# train
fig = plt.figure(figsize=(6,5))
ax = fig.add_subplot(111)
ax.plot(train_nam['Time'], train_nam['Observed'], 'g', label='Observed', linestyle = 'solid', linewidth = 2)
ax.plot(train_nam['Time'], train_nam['Simulated'], 'r', label='Simulated', linestyle = 'solid', linewidth = 2)

ax.legend()
ax.set_xlabel('Time')
#ax.set_ylabel('Value (m3/s)')

# test, train
original_train_test = original.loc[original['Time'] >= '2011-01-01']

fig = plt.figure(figsize=(10,4))
ax = fig.add_subplot(111)
ax.plot(original_train_test['Time'], original_train_test['d'], 'g', label='Observed', linestyle = 'solid', linewidth = 2)
ax.plot(train_nam['Time'], train_nam['Simulated'], 'r', label='Calibration', linestyle = 'solid', linewidth = 2)
ax.plot(predict_nam['Time'], predict_nam['Prediction'], 'y', label='Validation', linestyle = 'solid', linewidth = 2)

ax.legend()
ax.set_xlabel('Time')
ax.set_ylabel('Value (m3/s)')


############  Model Evaluation or performance Comparison with other models

################    Neural network - from the current code

#test

print(np.corrcoef(original_test['d'], test_prediction_d['d']))  #R
print(mean_absolute_error(original_test['d'], test_prediction_d['d'])) #MAE
print(sqrt(mean_squared_error(original_test['d'], test_prediction_d['d']))) #RMSE

#train
print(np.corrcoef(original_train['d'], train_prediction_d['d']))  #R
print(mean_absolute_error(original_train['d'], train_prediction_d['d'])) #MAE
print(sqrt(mean_squared_error(original_train['d'], train_prediction_d['d']))) #RMSE



###################   NAM Numerical modelling

#test
print(np.corrcoef(original_test['d'], predict_nam['Prediction']))  #R
print(mean_absolute_error(original_test['d'], predict_nam['Prediction'])) #MAE
print(sqrt(mean_squared_error(original_test['d'], predict_nam['Prediction']))) #RMSE

#train
print(np.corrcoef(original_train['d'], train_nam['Simulated']))  #R
print(mean_absolute_error(original_train['d'], train_nam['Simulated'])) #MAE
print(sqrt(mean_squared_error(original_train['d'], train_nam['Simulated']))) #RMSE


###################    Linear regression

from sklearn import linear_model

#Linear regression model
lr_model = linear_model.LinearRegression()
lr_model.fit(X_train_scaled, y_train)

test_pred_lr = lr_model.predict(X_test_scaled)
train_pred_lr = lr_model.predict(X_train_scaled[:1096])



#test
print(np.corrcoef(original_test['d'], test_pred_lr))  #R
print(mean_absolute_error(original_test['d'], test_pred_lr)) #MAE
print(sqrt(mean_squared_error(original_test['d'], test_pred_lr))) #RMSE

#train
print(np.corrcoef(original_train['d'], train_pred_lr))  #R
print(mean_absolute_error(original_train['d'],train_pred_lr)) #MAE
print(sqrt(mean_squared_error(original_train['d'], train_pred_lr))) #RMSE


##################    Decision tree
from sklearn.tree import DecisionTreeRegressor

tree = DecisionTreeRegressor()
tree.fit(X_train_scaled, y_train)

test_pred_tree = tree.predict(X_test_scaled)
train_pred_tree = tree.predict(X_train_scaled[:1096])

#test
print(sqrt(mean_squared_error(y_test, test_pred_tree )))
print(metrics.mean_absolute_error(y_test, test_pred_tree))
print(metrics.mean_squared_error(y_test, test_pred_tree))

#train
print(sqrt(mean_squared_error(y_train, train_pred_tree )))
print(metrics.mean_absolute_error(y_train, train_pred_tree))
print(metrics.mean_squared_error(y_train, train_pred_tree))

#test
print(np.corrcoef(original_test['d'], test_pred_tree))  #R
print(mean_absolute_error(original_test['d'], test_pred_tree)) #MAE
print(sqrt(mean_squared_error(original_test['d'], test_pred_tree))) #RMSE

#train
print(np.corrcoef(original_train['d'], train_pred_tree))  #R
print(mean_absolute_error(original_train['d'],train_pred_tree)) #MAE
print(sqrt(mean_squared_error(original_train['d'], train_pred_tree))) #RMSE



##################   Random forest

#Increase number of tress and see the effect
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators = 30, random_state=30)
model.fit(X_train_scaled, y_train)

test_pred_RF = model.predict(X_test_scaled)
train_pred_RF = model.predict(X_train_scaled[:1096])


#test
print(np.corrcoef(original_test['d'], test_pred_RF))  #R
print(mean_absolute_error(original_test['d'], test_pred_RF)) #MAE
print(sqrt(mean_squared_error(original_test['d'], test_pred_RF))) #RMSE

#train
print(np.corrcoef(original_train['d'],train_pred_RF))  #R
print(mean_absolute_error(original_train['d'],train_pred_RF)) #MAE
print(sqrt(mean_squared_error(original_train['d'], train_pred_RF))) #RMSE


#Feature ranking...
import pandas as pd
feature_list = list(X_test.columns)
feature_imp = pd.Series(model.feature_importances_, index=feature_list).sort_values(ascending=False)
print(feature_imp)

####################################################### all the three

fig = plt.figure(figsize=(10,4))
ax = fig.add_subplot(111)
ax.plot(original_train_test['Time'], original_train_test['d'], 'g', label='Observed', linestyle = 'solid', linewidth = 2)

ax.plot(test_prediction_d['Time'], test_prediction_d['d'], 'r', label='Prediction', linestyle = 'solid', linewidth = 2)
ax.plot(train_prediction_d['Time'], train_prediction_d['d'], 'r', label='Prediction', linestyle = 'solid', linewidth = 2)

ax.plot(predict_nam['Time'], predict_nam['Prediction'], 'b', label='Prediction', linestyle = 'solid', linewidth = 2)
ax.plot(train_nam['Time'], train_nam['Simulated'], 'b', label='Simulated', linestyle = 'solid', linewidth = 2)


ax.legend()
ax.set_xlabel('Time')
ax.set_ylabel('Value (m3/s)')

####################scatter plot

##test

##ANN
plt.figure(figsize=(6,5))
plt.scatter(original_test['d'], test_prediction_d['d'], c='black')
plt.plot([p1, p2], [p1, p2], 'r-')
plt.xlabel('Observed', fontsize=10)
plt.ylabel('Predictions', fontsize=10)
plt.title("ANN Validation ")
plt.show()




##NAM
plt.figure(figsize=(6,5))
plt.scatter(original_test['d'], predict_nam['Prediction'], c='black')
plt.plot([p1, p2], [p1, p2], 'r-')
plt.xlabel('Observed', fontsize=10)
#plt.ylabel('Predictions', fontsize=10)
plt.title("NAM Validation ")
plt.show()


##train

##ANN
plt.figure(figsize=(6,5))
plt.scatter(original_train['d'], train_prediction_d['d'], c='black')
plt.plot([p1, p2], [p1, p2], 'r-')
plt.xlabel('Observed', fontsize=10)
plt.ylabel('Predictions', fontsize=10)
plt.title("ANN Calibration ")
plt.show()




##NAM
plt.figure(figsize=(6,5))
plt.scatter(original_train['d'], train_nam['Simulated'], c='black')
plt.plot([p1, p2], [p1, p2], 'r-')
plt.xlabel('Observed', fontsize=10)
#plt.ylabel('Predictions', fontsize=10)
plt.title("NAM Calibration ")
plt.show()


