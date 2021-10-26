# -*- coding: utf-8 -*-
"""
Created on %(1st July 2021 )s

@author: %(Mumtaz Ahmad)s
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#directory
mydir = 'C:/Users/mumta/Desktop/SP/River Var Data/'

#reading saved file from Prepro
df = pd.read_csv(mydir+"Vesubei_all.csv" , parse_dates=[0],header=0,index_col=0 ,)

#meta data
Details = df.describe().transpose()


############################################ ploting and correlation

######################### Daily

######plotting

#Dischrage
fig = plt.figure(figsize=(8,3.5))
ax = fig.add_subplot(111)
ax.plot(df['d'], 'r', label='Dischrage', linestyle = 'solid', linewidth = 2)
ax.legend()
ax.set_xlabel('Time')
ax.set_ylabel('Value (m3/s)')

#Precipitation
fig = plt.figure(figsize=(8,3.5))
ax = fig.add_subplot(111)
ax.plot(df['p'], 'deepskyblue', label='Precipitation', linestyle = 'solid', linewidth = 2)
ax.legend()
ax.set_xlabel('Time')
ax.set_ylabel('Value (mm)')

#Temperature
fig = plt.figure(figsize=(8,3.5))
ax = fig.add_subplot(111)
ax.plot(df['t'], 'y', label='Temperature', linestyle = 'solid', linewidth = 2)
ax.legend()
ax.set_xlabel('Time')
ax.set_ylabel('Value (ᵒC)')

#Evaporation
fig = plt.figure(figsize=(8,3.5))
ax = fig.add_subplot(111)
ax.plot(df['e'], 'g', label='Evaporation', linestyle = 'solid', linewidth = 2)
ax.legend()
ax.set_xlabel('Time')
ax.set_ylabel('Value (mm)')


### correlation 
corrMatrixd = df.corr()


#matrix plot
plt.matshow(df.corr())
plt.xticks(range(len(df.columns)), df.columns)
plt.yticks(range(len(df.columns)), df.columns)
plt.colorbar()
plt.show()

# scater plot

plt.scatter(df['t'], df['e'])
plt.show()


#d vs p daily

fig, (tarh) = plt.subplots(figsize=(8,6), sharex=True)

tarh.plot(df['d'],color = 'r', linestyle = 'solid', linewidth = 2)
tarh.set_ylabel('Dischrage (m3/s)', color ='r', fontsize=15)
tarh.set_yticks([0,20,40,60,80,100])
tarh.tick_params(axis='y', labelcolor='r', labelsize=15)
tarh.patch.set_visible(False)

tarh2 = tarh.twinx()

tarh2.plot(df['p'], color='deepskyblue', linestyle = 'solid', linewidth = 2)
tarh2.set_ylabel('Precipitation (mm)', color ='deepskyblue', fontsize=15)
tarh2.set_yticks([0,25,50,75,100,125,150,175,200])
tarh2.tick_params(axis='y', labelcolor='deepskyblue', labelsize=15)
tarh2.set_zorder(tarh.get_zorder()-1)


#######################################    Monthly   

dfd_m = df['d'].resample('M').mean()
dfp_m = df['p'].resample('M').sum()
dft_m = df['t'].resample('M').mean()
dfe_m = df['e'].resample('M').mean()

df_m = pd.concat([dfd_m,dfp_m,dft_m,dfe_m ], axis=1)
 
#Dischrage
fig = plt.figure(figsize=(8,4))
ax = fig.add_subplot(111)
ax.plot(df_m['d'], 'g', label='Dischrage', linestyle = 'solid', linewidth = 2)
ax.legend()
ax.set_xlabel('Time')
ax.set_ylabel('Value (m3/s)')

#Precipitation
fig = plt.figure(figsize=(8,4))
ax = fig.add_subplot(111)
ax.plot(df_m['p'], 'r', label='Precipitation', linestyle = 'solid', linewidth = 2)
ax.legend()
ax.set_xlabel('Time')
ax.set_ylabel('Value (mm)')

#Temperature
fig = plt.figure(figsize=(8,4))
ax = fig.add_subplot(111)
ax.plot(df_m['t'], 'b', label='Temperature', linestyle = 'solid', linewidth = 2)
ax.legend()
ax.set_xlabel('Time')
ax.set_ylabel('Value (ᵒC)')

#Evaporation
fig = plt.figure(figsize=(8,4))
ax = fig.add_subplot(111)
ax.plot(df_m['e'], 'r', label='Evaporation', linestyle = 'solid', linewidth = 2)
ax.legend()
ax.set_xlabel('Time')
ax.set_ylabel('Value (mm)')


### correlation 
corrMatrixm = df_m.corr()

#matrix plot
plt.matshow(df_m.corr())
plt.xticks(range(len(df_m.columns)), df_m.columns)
plt.yticks(range(len(df_m.columns)), df_m.columns)
plt.colorbar()
plt.show()


# scater plot

plt.scatter(df_m['d'], df_m['p'])
plt.show()



#d vs p monthly

fig, (tarh) = plt.subplots(figsize=(8,6), sharex=True)

tarh.plot(df_m['d'],color = 'r', linestyle = 'solid', linewidth = 2)
tarh.set_ylabel('Dischrage (m3/s)', color ='r', fontsize=15)
tarh.set_yticks([0,5,10,15,20,25,30])
tarh.tick_params(axis='y', labelcolor='r', labelsize=15)
tarh.patch.set_visible(False)

tarh2 = tarh.twinx()

tarh2.plot(df_m['p'], color='deepskyblue', linestyle = 'solid', linewidth = 2)
tarh2.set_ylabel('Precipitation (mm)', color ='deepskyblue', fontsize=15)
tarh2.set_yticks([0,100,200,300,400,500,600])
tarh2.tick_params(axis='y', labelcolor='deepskyblue', labelsize=15)
tarh2.set_zorder(tarh.get_zorder()-1)





########################         correlation seasonal

def select_season(inDF, season, year=None):
    fake_idx = inDF.index + pd.DateOffset(months=1)   # --> python starts counting with 0 !!!
    seasons = {'DJF':1,'MAM':2,'JJA':3,'SON':4}
    if year is None:
        resultDF = inDF.groupby([fake_idx.quarter]).get_group((seasons[season]))
    else:
        resultDF = inDF.groupby([fake_idx.year,fake_idx.quarter]).get_group((year,seasons[season]))
    return resultDF

#DJF Winter ,JJA Summer , MAM Spring ,SON Autumn 

#DJF

d_DJF = select_season(df['d'],'DJF')
p_DJF = select_season(df['p'],'DJF')
t_DJF = select_season(df['t'],'DJF')
e_DJF = select_season(df['e'],'DJF')


#MAM

d_MAM = select_season(df['d'],'MAM')
p_MAM = select_season(df['p'],'MAM')
t_MAM = select_season(df['t'],'MAM')
e_MAM = select_season(df['e'],'MAM')

#JJA

d_JJA = select_season(df['d'],'JJA')
p_JJA = select_season(df['p'],'JJA')
t_JJA = select_season(df['t'],'JJA')
e_JJA = select_season(df['e'],'JJA')


#SON

d_SON = select_season(df['d'],'SON')
p_SON = select_season(df['p'],'SON')
t_SON = select_season(df['t'],'SON')
e_SON = select_season(df['e'],'SON')



df_s = pd.concat([d_DJF,d_MAM,d_JJA,d_SON,
                  p_DJF,p_MAM,p_JJA,p_SON,
                  e_DJF,e_MAM,e_JJA,t_SON,
                  t_DJF,t_MAM,t_JJA,e_SON], axis=1)

### correlation 
corrMatrixs = df_s.corr()

#matrix plot
plt.matshow(df_s.corr())
plt.xticks(range(len(df_s.columns)), df_s.columns)
plt.yticks(range(len(df_s.columns)), df_s.columns)
plt.colorbar()
plt.show()


# d & p SON .49 corr

# t DJF & e SON .68
# t JAJ & e MAM .76
# t SON & e JAJ .72

#d vs p SON



fig, (tarh) = plt.subplots(figsize=(8,6), sharex=True)

tarh.plot(d_SON['2014-09-01 00:00:00':'2014-11-30 00:00:00'],color = 'r', linestyle = 'solid', linewidth = 2)
tarh.set_ylabel('Dischrage (m3/s)', color ='r', fontsize=15)
tarh.set_yticks([0,15,30,45,60,75,90,105])
tarh.tick_params(axis='y', labelcolor='r', labelsize=15)
tarh.patch.set_visible(False)

tarh2 = tarh.twinx()

tarh2.plot(p_SON['2014-09-01 00:00:00':'2014-11-30 000:00:00'], color='deepskyblue', linestyle = 'solid', linewidth = 2)
tarh2.set_ylabel('Precipitation (mm)', color ='deepskyblue', fontsize=15)
tarh2.set_yticks([0,50,100,150,200])
tarh2.tick_params(axis='y', labelcolor='deepskyblue', labelsize=15)
tarh2.set_zorder(tarh.get_zorder()-1)








