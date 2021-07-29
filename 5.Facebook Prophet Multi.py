# -*- coding: utf-8 -*-
"""
Created on %(1st July 2021 )s

@author: %(Mumtaz Ahmad)s
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from fbprophet import Prophet


mydir = 'C:/Users/mumta/Desktop/SP/River Var Data/'



#all in one

df= pd.read_csv(mydir+"Visubei2.csv" , parse_dates=[0],header=0,)



Details = df.describe().transpose()

df.plot(subplots=True);



data = df[['Time', 'Discharge(m3/s)','Precipitation(mm)']] 
data.columns = ['ds', 'y','preci',] 

#data['month']=data['ds'].dt.month

####
data[['y', 'preci']].corr()
data.query('preci>100')[['y', 'preci']].corr()

data.query('preci<100')[['y', 'preci']].corr()

###


def autumn_preci(preci):
  if preci > 50:
    return 1
  else:
    return 0


data['autumn_preci']=data['preci'].apply(autumn_preci)

data['month_bins']=pd.cut(data['month'],bins=3,
                                 labels=False)




###
train= data.iloc[:1096]
test= data.iloc[1096:]


model = Prophet(interval_width=0.95, yearly_seasonality=True)



model.add_regressor('autumn_preci', standardize=False)
model.add_regressor('month_bins', standardize=False, mode='multiplicative')


model.fit(train)


model.params


future = model.make_future_dataframe(periods=365)



future['autumn_preci'] = data['autumn_preci']
future['month_bins'] = data['month_bins']


future

forecast = model.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

fig1 = model.plot(forecast)



fig2 = model.plot_components(forecast)


from fbprophet.diagnostics import performance_metrics, cross_validation
cv_results = cross_validation(model=model, initial='731 days', horizon='365 days')
df_p = performance_metrics(cv_results)
df_p


from fbprophet.plot import plot_cross_validation_metric
fig3 = plot_cross_validation_metric(cv_results, metric='mape')



















