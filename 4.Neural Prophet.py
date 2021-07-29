# -*- coding: utf-8 -*-
"""
Created on %(1st July 2021 )s

@author: %(Mumtaz Ahmad)s
"""

pip install numpy
pip install skipy
#conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch

!pip install neuralprophet
from neuralprophet import NeuralProphet



import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import pickle


mydir = 'C:/Users/mumta/Desktop/SP/River Var Data/'

df = pd.read_csv(mydir+"Daily_Discharge_update.csv" , parse_dates=[0],header=0,)



#train= df.iloc[:1096]
#test= df.iloc[1096:]


data = df[['Time', 'Discharge(m3/s)']] 
data.columns = ['ds', 'y'] 
data.head()

#Train Model
m = NeuralProphet()
model = m.fit(data, freq='D', epochs=100)

#Forecast Away
future = m.make_future_dataframe(data, periods=365)
forecast = m.predict(future)
forecast.head()

plot1 = m.plot(forecast)
plt2 = m.plot_components(forecast)


#validation



ax = forecast.plot()
df.plot(ax=ax)

import seaborn as sns

sns.lineplot(df['Time'], df['Discharge(m3/s)'])
sns.lineplot(forecast['ds'], forecast['yhat1'])





