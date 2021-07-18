# -*- coding: utf-8 -*-
"""
Created on Thu May 27 18:08:06 2021

@author: mumta
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



mydir = 'C:/Users/mumta/Desktop/SP/River Var Data/'

df1 = pd.read_csv(mydir+"Daily_Discharge.csv" , parse_dates=[0],header=0,index_col=0,)

df2 = pd.read_csv(mydir+"Daily_Precipitation.csv" , parse_dates=[0],header=0, index_col=0)

df3 = pd.read_csv(mydir+"Daily_Temperature.csv" , parse_dates=[0], header=0, index_col=0 )

df4 = pd.read_csv(mydir+"Daily_Evaporation.csv" , parse_dates=[0], header=0, index_col=0 )


#all in one

#df_all= pd.read_csv(mydir+"Visubei.csv" , parse_dates=[0],header=0,index_col=0 ,)
#Details = df_all.describe().transpose()


#Extracting years data 2011-14

df11 = df1 
df22 = df2['2011-01-01 00:00:00':'2014-12-31 00:00:00']
df33 = df3['2011-01-01 00:00:00':'2014-12-31 00:00:00']
df44 = df4['2011-01-01 00:00:00':'2014-12-31 00:00:00']


dfd = pd.DataFrame({'Value':np.array(df11['Discharge(m3/s)'])}, index=df11.index)
dfp = pd.DataFrame({'Value':np.array(df22['Precipitation(mm)'])}, index=df22.index)
dft = pd.DataFrame({'Value':np.array(df33['Mean Temperature(C)'])}, index=df33.index)
dfe = pd.DataFrame({'Value':np.array(df44['Evaporation'])}, index=df44.index)



#Calculate monthly mean or sum values

dfd_monthly = dfd.resample('M').mean()
dfp_monthly = dfp.resample('M').sum()
dft_monthly = dft.resample('M').mean()
dfe_monthly = dfe.resample('M').mean()

#Checking NA

dfd.isnull().sum()
dfp.isnull().sum()
dft.isnull().sum()
dfe.isnull().sum()


#To check the negative data

dfd.index[dfd['Value'] < 0]
dfp.index[dfp['Value'] < 0]



#There is some negative value in discharge in may 2013 and december 2015

#for may 2013

dis = dfd['2013-05-01 00:00:00':'2013-05-31 00:00:00']['Value']
bad_dis = dis == -1e-30
dis[bad_dis] = 23.8  

#for december 2015

dis = dfd['2014-12-01 00:00:00':'2014-12-31 00:00:00']['Value']
bad_dis = dis == -1e-30
dis[bad_dis] = 10.2



#Calculate the mean, max of discharge,precipitation and temperature

dfd.describe().apply(lambda s: s.apply('{0:.5f}'.format))
dfp.describe().apply(lambda s: s.apply('{0:.5f}'.format))
dft.describe().apply(lambda s: s.apply('{0:.5f}'.format))
dfe.describe().apply(lambda s: s.apply('{0:.5f}'.format))


#plottting

dfd.plot()
dfp.plot()
dft.plot()
dfe.plot()

###Correlation

#b/w discharge and precipitation

np.corrcoef(dfd['Value'], dfp['Value'])
np.corrcoef(dfd_monthly['Value'], dfp_monthly['Value']) #(best correlation .66)

#b/w discharge and temperature

np.corrcoef(dfd['Value'], dft['Value'])
np.corrcoef(dfd_monthly['Value'], dft_monthly['Value'])

#b/w discharge and evaporation

np.corrcoef(dfd['Value'], dfe['Value'])
np.corrcoef(dfd_monthly['Value'], dfe_monthly['Value'])


#4Delineate the data into the 4 seasons: DJF, MAM, JJA, SON.

#discharge
season = ((dfd.index.month % 12 + 3) // 3).map({1:'DJF', 2: 'MAM', 3:'JJA', 4:'SON'})

dfd_DJF = dfd[season == 'DJF']
dfd_MAM = dfd[season == 'MAM']
dfd_JJA = dfd[season == 'JJA']
dfd_SON = dfd[season == 'SON']

#precipitation
season = ((dfp.index.month % 12 + 3) // 3).map({1:'DJF', 2: 'MAM', 3:'JJA', 4:'SON'})

dfp_DJF = dfp[season == 'DJF']
dfp_MAM = dfp[season == 'MAM']
dfp_JJA = dfp[season == 'JJA']
dfp_SON = dfp[season == 'SON']

#temperature
season = ((dft.index.month % 12 + 3) // 3).map({1:'DJF', 2: 'MAM', 3:'JJA', 4:'SON'})

dft_DJF = dft[season == 'DJF']
dft_MAM = dft[season == 'MAM']
dft_JJA = dft[season == 'JJA']
dft_SON = dft[season == 'SON']

#evaporation
season = ((dfe.index.month % 12 + 3) // 3).map({1:'DJF', 2: 'MAM', 3:'JJA', 4:'SON'})

dfe_DJF = dfe[season == 'DJF']
dfe_MAM = dfe[season == 'MAM']
dfe_JJA = dfe[season == 'JJA']
dfe_SON = dfe[season == 'SON']


###seasonal correlation

#b/w discharge and precipitation 

np.corrcoef(dfd_DJF['Value'], dfp_DJF['Value'])
np.corrcoef(dfd_MAM['Value'], dfp_MAM['Value'])
np.corrcoef(dfd_JJA['Value'], dfp_JJA['Value'])
np.corrcoef(dfd_SON['Value'], dfp_SON['Value']) #(best correlation .49)

#b/w discharge and temperature

np.corrcoef(dfd_DJF['Value'], dft_DJF['Value'])
np.corrcoef(dfd_MAM['Value'], dft_MAM['Value'])
np.corrcoef(dfd_JJA['Value'], dft_JJA['Value'])
np.corrcoef(dfd_SON['Value'], dft_SON['Value'])


#b/w discharge and evaporation

np.corrcoef(dfd_DJF['Value'], dfe_DJF['Value'])
np.corrcoef(dfd_MAM['Value'], dfe_MAM['Value'])
np.corrcoef(dfd_JJA['Value'], dfe_JJA['Value'])
np.corrcoef(dfd_SON['Value'], dfe_SON['Value'])
