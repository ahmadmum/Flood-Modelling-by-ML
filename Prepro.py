# -*- coding: utf-8 -*-
"""
Created on Thu May 27 18:08:06 2021

@author: mumta
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



mydir = 'C:/Users/mumta/Desktop/SP/Data/'

dfd = pd.read_csv(mydir+"Daily_Discharge.csv" , parse_dates=[0],
                 header=0,index_col=0)

df1 = pd.read_csv(mydir+"Daily_Precipitation.csv" , parse_dates=[0],
                 header=0, index_col=0)

df2 = pd.read_csv(mydir+"Daily_Temperature.csv" , parse_dates=[0],
                 header=0, index_col=0 )

#Extracting years data 2011-14

dfp = df1['2011-01-01 07:00:00':'2014-12-31 07:00:00']
dft = df2['2011-01-01 07:00:00':'2014-12-31 07:00:00']


#Calculate the mean, max, min of discharge

mean_dis = dfd.mean()


max_dis = dfd.max()
max_dis_date = dfd.index.max()


mean_pre = dfp.mean()
mean_tem = dft.mean()

#Calculate the max value

max_dis = dfd.max()

max_pre = dfp.max()
max_pre = dfp.index.max()

#Calculate the sum

sum_pre = dfp.sum()

#Checking NA

dfd.isnull().sum()
dfp.isnull().sum()
dft.isnull().sum()

#To check the negative data

neg_dis = dfd.index[dfd['Vesubie(m3/s)'] < 0]
neg_pre = dfp.index[dfp['Precipitation(mm)'] < 0]

#plottting

dfd.plot()
dfp.plot()
dft.plot()

df1.plot()


#ANN Library

#keras : Tensorflow , Theano , CNTK