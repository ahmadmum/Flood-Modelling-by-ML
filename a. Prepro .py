# -*- coding: utf-8 -*-
"""
Created on Thu May 27 18:08:06 2021

@author: mumta
"""

import pandas as pd
import numpy as np
import statistics

#directory
mydir = 'C:/Users/mumta/Desktop/SP/River Var Data/'

#reading file
df_d = pd.read_csv(mydir+"Discharge.csv" , parse_dates=[0],header=0,index_col=0,)
df_p = pd.read_csv(mydir+"Precipitation.csv" , parse_dates=[0],header=0, index_col=0)
df_t = pd.read_csv(mydir+"Daily_Temperature.csv" , parse_dates=[0], header=0, index_col=0 )
df_e = pd.read_csv(mydir+"Evaporation.csv" , parse_dates=[0], header=0, index_col=0 )

#extracting years data 2011-14
dfd = df_d #(already four years)
dfp = df_p['2011-01-01 00:00:00':'2014-12-31 00:00:00']
dft = df_t['2011-01-01 00:00:00':'2014-12-31 00:00:00']
dfe = df_e['2011-01-01 00:00:00':'2014-12-31 00:00:00']


#meta data
Details_d = dfd.describe().transpose()
Details_p = dfp.describe().transpose()
Details_t = dft.describe().transpose()
Details_e = dfe.describe().transpose()

#To check the negative date of discharge
dfd.index[dfd['Discharge(m3/s)'] < 0]


#There is some negative value in discharge in may 2013 and december 2015
#for may 2013-05-06 to 2013-05-10
dfd = dfd['2011-01-01 00:00:00':'2014-12-31 00:00:00']['Discharge(m3/s)']
bad_dism = dfd == -1e-30
dfd[bad_dism] = statistics.mean(dfd['2013-05-03 00:00:00':'2013-05-05 00:00:00'])

#for december 2014-12-14 to 2014-12-16
bad_disd = dfd == -1e-30
dfd[bad_disd] = statistics.mean(dfd['2014-12-10 00:00:00':'2014-12-13 00:00:00'])


#all parameter in one file ,
df_all = pd.concat([dfd,dfp,dft,dfe ], axis=1,)

df_all = df_all.rename(columns={'Discharge(m3/s)':'d', 'Precipitation(mm)':'p', 'Mean Temperature(C)':'t', 'Evaporation':'e', })

#deatils of all
Details_all = df_all.describe().transpose()

#saving file
df_all.to_csv (r'C:\Users\mumta\Desktop\SP\River Var data/Vesubei_all.csv',index=True,  header=True)










