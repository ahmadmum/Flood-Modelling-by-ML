# -*- coding: utf-8 -*-
"""
Created on %(1st July 2021 )s

@author: %(Mumtaz Ahmad)s
"""

import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Dropout
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.metrics import mean_squared_error
from math import sqrt



mydir = 'C:/Users/mumta/Desktop/SP/River Var Data/'

df = pd.read_csv(mydir+"Daily_Discharge_update.csv" , parse_dates=[0],header=0,)

# know the lenth
len(dfd)

#Separate dates for future plotting

train= df.iloc[:1096]
test= df.iloc[1096:]

train_dates = pd.to_datetime(train['Time'])



#Variables for training

cols = list(train)[1:2]
df_for_training = train[cols].astype(float)



# df_for_plot=df_for_training.tail(5000)
# df_for_plot.plot.line()

#LSTM uses sigmoid and tanh that are sensitive to magnitude so values need to be normalized
# normalize the dataset
scaler = StandardScaler()
scaler = scaler.fit(df_for_training)
df_for_training_scaled = scaler.transform(df_for_training)


#As required for LSTM networks, we require to reshape an input data into n_samples x timesteps x n_features. 
#In this example, the n_features is 2. We will make timesteps = 3. 
#With this, the resultant n_samples is 5 (as the input data has 9 rows).
trainX = []
trainY = []

n_future = 1   # Number of days we want to predict into the future
n_past = 14     # Number of past days we want to use to predict the future

for i in range(n_past, len(df_for_training_scaled) - n_future +1):
    trainX.append(df_for_training_scaled[i - n_past:i, 0:df_for_training.shape[1]])
    trainY.append(df_for_training_scaled[i + n_future - 1:i + n_future, 0])

trainX, trainY = np.array(trainX), np.array(trainY)



print('trainX shape == {}.'.format(trainX.shape))
print('trainY shape == {}.'.format(trainY.shape))


# define Autoencoder model

model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
model.add(LSTM(32, activation='relu', return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(trainY.shape[1]))

model.compile(optimizer='adam', loss='mse')
model.summary()


# fit model
history = model.fit(trainX, trainY, epochs=10, batch_size=16, validation_split=0.1, verbose=1)

#plt.plot(history.history['loss'], label='Training loss')
#plt.plot(history.history['val_loss'], label='Validation loss')
#plt.legend()

#Forecasting...
#Start with the last day in training date and predict future...
n_future=365  #Redefining n_future to extend prediction dates beyond original n_future dates...
forecast_period_dates = pd.date_range(list(train_dates)[-1], periods=n_future, freq='1d').tolist()

forecast = model.predict(trainX[-n_future:]) #forecast 


#Perform inverse transformation to rescale back to original range
#Since we used 5 variables for transform, the inverse expects same dimensions
#Therefore, let us copy our values 5 times and discard them after inverse transform
forecast_copies = np.repeat(forecast, df_for_training.shape[1], axis=-1)
y_pred_future = scaler.inverse_transform(forecast_copies)[:,0]


# Convert timestamp to date
forecast_dates = []
for time_i in forecast_period_dates:
    forecast_dates.append(time_i.date())
    
df_forecast = pd.DataFrame({'Time':np.array(forecast_dates), 'Discharge(m3/s)':y_pred_future})
df_forecast['Time']=pd.to_datetime(df_forecast['Time'])


# comparing forecasting data with testing data (df_forecast) & (test)

matplotlib.style.use('ggplot')
plt.scatter(df_forecast['Discharge(m3/s)'], test['Discharge(m3/s)'])
plt.show()

sns.lineplot(test['Time'], test['Discharge(m3/s)'])
sns.lineplot(df_forecast['Time'], df_forecast['Discharge(m3/s)'])




#validation

#rmse

rmse=sqrt(mean_squared_error(df_forecast['Discharge(m3/s)'], test['Discharge(m3/s)']))
print(rmse)

#corr

np.corrcoef(df_forecast['Discharge(m3/s)'], test['Discharge(m3/s)'])

#mean_absolute_error

from sklearn.metrics import mean_absolute_error
mean_absolute_error(df_forecast['Discharge(m3/s)'], test['Discharge(m3/s)'])







