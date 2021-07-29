# -*- coding: utf-8 -*-
"""
Created on %(1st July 2021 )s

@author: %(Mumtaz Ahmad)s
"""


import warnings; 
warnings.simplefilter('ignore')

!pip install pystan
!pip install fbprophet




import pandas as pd
from fbprophet import Prophet


mydir = 'C:/Users/mumta/Desktop/SP/River Var Data/'

df = pd.read_csv(mydir+"Daily_Discharge_update.csv" , parse_dates=[0],header=0,)

data = df[['Time', 'Discharge(m3/s)']] 
data.columns = ['ds', 'y'] 

# Train Model


m = Prophet(interval_width=0.95, daily_seasonality=True)
model = m.fit(data)

future = m.make_future_dataframe(periods=365,freq='D')
forecast = m.predict(future)
forecast.head()


plot1 = m.plot(forecast)



