# -*- coding: utf-8 -*-
"""
Created on Thu May 27 18:08:06 2021

@author: mumtaz
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



mydir = 'C:/Users/mumta/Desktop/SP/River Var Data/'

df1 = pd.read_csv(mydir+"Daily_Discharge.csv" , parse_dates=[0],header=0,index_col=0,)

df2 = pd.read_csv(mydir+"Daily_Precipitation.csv" , parse_dates=[0],header=0, index_col=0)

df3 = pd.read_csv(mydir+"Daily_Temperature.csv" , parse_dates=[0], header=0, index_col=0 )

df4 = pd.read_csv(mydir+"Daily_Evaporation.csv" , parse_dates=[0], header=0, index_col=0 )

#Extracting years data 2011-14

df11 = df1 
df22 = df2['2011-01-01 07:00:00':'2014-12-31 07:00:00']
df33 = df3['2011-01-01 07:00:00':'2014-12-31 12:00:00']
df44 = df4['2011-01-01 07:00:00':'2014-12-31 07:00:00']


dfd = pd.DataFrame({'Value':np.array(df11['Vesubie(m3/s)'])}, index=df11.index)
dfp = pd.DataFrame({'Value':np.array(df22['Precipitation(mm)'])}, index=df22.index)
dft = pd.DataFrame({'Value':np.array(df33['Mean Temperature(C)'])}, index=df33.index)
dfe = pd.DataFrame({'Value':np.array(df44['Evaporation'])}, index=df44.index)


#Calculate the mean, max of discharge,precipitation and temperature

dfd.describe().apply(lambda s: s.apply('{0:.5f}'.format))
dfp.describe().apply(lambda s: s.apply('{0:.5f}'.format))
dft.describe().apply(lambda s: s.apply('{0:.5f}'.format))
dfe.describe().apply(lambda s: s.apply('{0:.5f}'.format))


#Checking NA

dfd.isnull().sum()
dfp.isnull().sum()
dft.isnull().sum()
dfe.isnull().sum()


#To check the negative data

dfd.index[dfd['Value'] < 0]
dfp.index[dfp['Value'] < 0]
dft.index[dfd['Value'] < 0]
dfe.index[dfp['Value'] < 0]

#plottting

dfd.plot()
dfp.plot()
dft.plot()
dfe.plot()

###Correlation

#b/w discharge and precipitation

np.corrcoef(dfd['Value'], dfp['Value'])

#b/w discharge and temperature

np.corrcoef(dfd['Value'], dft['Value'])

#b/w discharge and evaporation

np.corrcoef(dfd['Value'], dfe['Value'])


#4Delineate the data into the 4 seasons: DJF, MAM, JJA, SON.

#discharge

season = ((dfd.index.month % 12 + 3) // 3).map({1:'DJF', 2: 'MAM', 3:'JJA', 4:'SON'})

dfd_DJF = dfd[season == 'DJF']
dfd_MAM = dfd[season == 'MAM']
dfd_JJA = dfd[season == 'JJA']
dfd_SON = dfd[season == 'SON']

season = ((dfp.index.month % 12 + 3) // 3).map({1:'DJF', 2: 'MAM', 3:'JJA', 4:'SON'})

dfp_DJF = dfp[season == 'DJF']
dfp_MAM = dfp[season == 'MAM']
dfp_JJA = dfp[season == 'JJA']
dfp_SON = dfp[season == 'SON']


season = ((dft.index.month % 12 + 3) // 3).map({1:'DJF', 2: 'MAM', 3:'JJA', 4:'SON'})

dft_DJF = dft[season == 'DJF']
dft_MAM = dft[season == 'MAM']
dft_JJA = dft[season == 'JJA']
dft_SON = dft[season == 'SON']

season = ((dfe.index.month % 12 + 3) // 3).map({1:'DJF', 2: 'MAM', 3:'JJA', 4:'SON'})

dfe_DJF = dfe[season == 'DJF']
dfe_MAM = dfe[season == 'MAM']
dfe_JJA = dfe[season == 'JJA']
dfe_SON = dfe[season == 'SON']



###Correlation


#b/w discharge and precipitation

np.corrcoef(dfd_DJF['Value'], dfp_DJF['Value'])
np.corrcoef(dfd_MAM['Value'], dfp_MAM['Value'])
np.corrcoef(dfd_JJA['Value'], dfp_JJA['Value'])
np.corrcoef(dfd_SON['Value'], dfp_SON['Value'])




#ANN Library

#keras : Tensorflow , Theano , CNTK, neuralprophet


import numpy
import matplotlib.pyplot as plt
import pandas
conda install keras

from keras.models import Sequential
from keras.layers import Dense
