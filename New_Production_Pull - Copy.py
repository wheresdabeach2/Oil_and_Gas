#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import datetime as dt
import numpy as np
import random
from scipy.optimize import curve_fit


# Read in worksheet from website https://www.depgreenport.state.pa.us/ReportExtracts/OG/OilGasWellProdReport
# for All wells and save to location for file
# saved_file is the directory location of where the file will be saved

# In[ ]:


#file = (insert file here as csv
#saved_file = (insert final save location here as csv)


# In[ ]:


data = pd.read_csv(file)

data_preserved = data


# In[ ]:


#Remove inactive wells
data = data[data['WELL_STATUS'] == 'Active']


# In[ ]:


#Remove averaged production wells
data = data[data['AVERAGED'] == 'No']


# In[ ]:


#Convert string dates to datetimes
data['SPUD_DATE'] = pd.to_datetime(data['SPUD_DATE'])
data['PRODUCTION_PERIOD_START_DATE'] = data['PRODUCTION_PERIOD_START_DATE'].apply(lambda x: dt.datetime.strptime(x,'%m/%d/%Y'))
data['PRODUCTION_PERIOD_END_DATE'] = data['PRODUCTION_PERIOD_END_DATE'].apply(lambda x: dt.datetime.strptime(x,'%m/%d/%Y'))


# In[ ]:


#Use time intervals to calculate total days
data['TOTAL_DAYS'] = data['PRODUCTION_PERIOD_END_DATE'].sub(data['PRODUCTION_PERIOD_START_DATE'], axis=0)
data['TOTAL_DAYS'] = data['TOTAL_DAYS'] / np.timedelta64(1, 'D')

#if there are no production days filled, assume full time period production, otherwise given production days
data.GAS_PRODUCTION_DAYS.fillna(data.TOTAL_DAYS, inplace=True)
#Delete Production days
del data['TOTAL_DAYS']


# In[ ]:


#Fix wells that have innacurate Days for first month's production (3 days should be 30)
bad_range_IP_apis = ['059-27077', '059-27078', '059-27079', '059-27080']
for i in bad_range_IP_apis:
    data.loc[(data['GAS_PRODUCTION_DAYS'] ==3) &  (data['PERIOD_ID'] =='17NOVP') & (data['WELL_PERMIT_NUM'] == i), 'GAS_PRODUCTION_DAYS'] = 30


# In[ ]:


#Calculate average production per day producing period
data['AVERAGE_GAS_DAILY_PRODUCTION'] = data['GAS_QUANTITY']/data['GAS_PRODUCTION_DAYS']


# In[ ]:


#Remove conventional wells, vertical wells and negative, 50 or infinite values in the production
data_unconv = data[(data['AVERAGE_GAS_DAILY_PRODUCTION']>50) &(data['AVERAGE_GAS_DAILY_PRODUCTION']<10000000000000)]
data_unconv = data_unconv[data_unconv['UNCONVENTIONAL'] == 'Yes']
data_unconv = data_unconv[data_unconv['WELL_CONFIGURATION'] == 'Horizontal Well']


# In[ ]:


#Sort data by permit num then production date
data_unconv = data_unconv.sort_values(['WELL_PERMIT_NUM', 'PRODUCTION_PERIOD_START_DATE'])


# In[ ]:


#Remove wells that have no operator
data_unconv.drop(data_unconv[data_unconv['OPERATOR'] == ''].index, inplace = True)


# In[ ]:


#Function - Reset index, find index where gas production is maximum, return df starting at max index
def FindMax(df):
    df = df.reset_index(drop = True)
    maxin = df['AVERAGE_GAS_DAILY_PRODUCTION'].idxmax()
    df = df[maxin:]
    return df


# In[ ]:


#Apply FindMax to data to remove values before max production
data_unconv_max = data_unconv.groupby('WELL_PERMIT_NUM').apply(FindMax)


# In[ ]:


#Recategorize Rice and Chevron wells to EQT wells
data_unconv_max['OPERATOR'] = data_unconv_max['OPERATOR'].replace(['RICE DRILLING B LLC', 'CHEVRON APPALACHIA LLC'], 'EQT PROD CO')


# In[ ]:


#Add IP Column
data_unconv_max.reset_index(inplace=True, drop=True)
data_unconv_max['IP'] = data_unconv_max.groupby('WELL_PERMIT_NUM')['AVERAGE_GAS_DAILY_PRODUCTION'].transform('max')


# In[ ]:


#Add months of production column
data_unconv_max.reset_index(inplace=True, drop=True)
data_unconv_max['Months'] = data_unconv_max.groupby(['WELL_PERMIT_NUM'])['GAS_PRODUCTION_DAYS'].apply(lambda x: x.cumsum()/30.4)   


# In[ ]:


#Add TIL year
data_unconv_max['YEAR_TIL'] = data_unconv_max.groupby(['WELL_PERMIT_NUM'])['PRODUCTION_PERIOD_START_DATE'].transform(min)
data_unconv_max['YEAR_TIL'] = pd.DatetimeIndex(data_unconv_max['YEAR_TIL']).year


# In[ ]:


#Remove Permit nums that have less than 12 lines (production data points)
data_unconv_max = data_unconv_max.groupby('WELL_PERMIT_NUM').filter(lambda x : len(x)>12)


# In[ ]:


len(data_unconv_max)


# In[ ]:


#Drop unused columns
unused_columns = ['PERIOD_ID', 'REPORTING_PERIOD',
       'PRODUCTION_INDICATOR', 'WELL_STATUS','OIL_QUANTITY',
       'OIL_PRODUCTION_DAYS', 'AVERAGED', 'GROUP_NO', 'OGO_NUM',
       'UNCONVENTIONAL', 'WELL_CONFIGURATION',
       'NON_PRODUCTION_COMMENTS', 'ADDITIONAL_COMMENTS', 'REPORT_GENERATED_DATE', 'RECORD_SOURCE',
                'WELL_TYPE']

data_unconv_max = data_unconv_max.drop(columns = unused_columns)


# In[ ]:


#Plot data for random individual well
permits = data_unconv_max.WELL_PERMIT_NUM.unique().tolist()
import plotly.graph_objects as go

api = '035-21178'
print(api)
xwell = data_unconv_max[data_unconv_max.WELL_PERMIT_NUM == api]['PRODUCTION_PERIOD_START_DATE']
ywell = data_unconv_max[data_unconv_max.WELL_PERMIT_NUM == api]['AVERAGE_GAS_DAILY_PRODUCTION']

fig = go.Figure()
fig.add_trace(go.Scatter(
                    x=xwell,
                    y=ywell,
                    mode='lines',
                    name='Daily Production'))

fig.show()


# In[ ]:


#Function to determine Arps curve production
def decline_curve(qi):
    def Arps(t,di,b):
        return qi / np.power((1+b*di*t),1./b)
    return Arps


# In[ ]:


#Provides forecast for row with arps variables
def get_forecast(row):
    d_i = row.d_i
    b = row.b
    q_i =row.q_i
    t = row.Months
    
    return q_i / np.power((1+b*d_i*t),1./b)


# In[ ]:


#Create Arps values and forecast for dataframe
#Error API's initiliaze
errors =[]
def Projection(df):
    df.reset_index(inplace=True, drop=True)
    t = df.Months
    q = df.AVERAGE_GAS_DAILY_PRODUCTION
    #Initial Production
    hyp = decline_curve(q[0])
    p0 = [.5,1]
    try:
        x,y = curve_fit(hyp, t,q, maxfev = 10000, p0=p0)
    except:
        return errors.append(df['WELL_PERMIT_NUM'].iloc[1])
    d_i = x[0]
    b = x[1]
    df['d_i'] = d_i
    df['b'] = b
    df['q_i'] = q[0]
    df['FORECAST'] = df.apply(get_forecast, axis=1)
    return df


# In[ ]:


#reset index and sort values
data_unconv_max.reset_index(inplace=True, drop=True)
data_unconv_max = data_unconv_max.sort_values(['WELL_PERMIT_NUM', 'PRODUCTION_PERIOD_START_DATE'])


# In[ ]:


#run projection on full dataset
fullset = data_unconv_max
fullset = fullset.groupby('WELL_PERMIT_NUM').apply(Projection)


# In[ ]:


#Drop wells with high or low b values and high Di values
fullset.drop(fullset[fullset['b'] > 3].index, inplace=True)


# In[ ]:


fullset.drop(fullset[fullset['b'] < 0].index, inplace=True)


# In[ ]:


fullset.drop(fullset[fullset['d_i'] > 1].index, inplace=True)


# In[ ]:


#Add TIL year
fullset.reset_index(inplace=True, drop=True)
fullset['YEAR_TIL'] = pd.DatetimeIndex(fullset.groupby(['WELL_PERMIT_NUM'])['PRODUCTION_PERIOD_START_DATE'].transform(min)).year


# In[ ]:


#Extend each well into the future for production start, end, and months columns while duplicating 
#the rest of the columns
def increase_dates(df):
    periods = 96    #Number of months to extend forecast out
    df.reset_index(inplace=True, drop=True)
    df = df.sort_values(['WELL_PERMIT_NUM', 'PRODUCTION_PERIOD_START_DATE'])
    starter = df[['PRODUCTION_PERIOD_START_DATE']]
    starter = starter.append(pd.DataFrame({'PRODUCTION_PERIOD_START_DATE': pd.date_range(start=starter.PRODUCTION_PERIOD_START_DATE.iloc[-1], periods= periods, freq='M',closed='right')}))
    ender = df[['PRODUCTION_PERIOD_END_DATE']]
    ender = ender.append(pd.DataFrame({'PRODUCTION_PERIOD_END_DATE': pd.date_range(start=ender.PRODUCTION_PERIOD_END_DATE.iloc[-1], periods= (periods+1), freq='M',closed='right')}))
    months = df[['Months']]
    max_months = max(months['Months'])
    for i in range(periods):
        months = months.append(pd.DataFrame({'Months': i+max_months+1}, index=[0]), ignore_index=True)
    starter = starter.reset_index(drop=True)
    ender = ender.reset_index(drop=True)
    months = months.reset_index(drop=True)
    df = df.drop(columns = ['PRODUCTION_PERIOD_START_DATE', 'PRODUCTION_PERIOD_END_DATE', 'Months', 'FORECAST'], axis =1)
    df = pd.concat([df,starter], axis=1)
    df = pd.concat([df, ender], axis =1)
    df = pd.concat([df, months], axis =1)
    columns = ['WELL_PERMIT_NUM', 'FARM_NAME_WELL_NUM', 'SPUD_DATE', 'GAS_QUANTITY',
       'GAS_PRODUCTION_DAYS', 'CONDENSATE_QUANTITY',
       'CONDENSATE_PRODUCTION_DAYS', 'OPERATOR', 'WELL_COUNTY',
       'WELL_MUNICIPALITY', 'WELL_LATITUDE', 'WELL_LONGITUDE',
       'SUBMISSION_FINAL_DATE', 'IP',
       'YEAR_TIL', 'd_i', 'b', 'q_i']
    for y in columns:
        df[y] = df[y].iloc[1]
    df['FORECAST_OUT'] = df.apply(get_forecast, axis =1)
    return df


# In[ ]:


#Run extended well to get future date and forecast values for the full dataset
fullset.reset_index(inplace=True, drop = True)
ext_fullset = fullset.groupby('WELL_PERMIT_NUM').apply(increase_dates)


# In[ ]:


ext_fullset.reset_index(inplace=True, drop=True)


# In[ ]:


#Plot data for random individual well
permits = ext_fullset.WELL_PERMIT_NUM.unique().tolist()
api = random.choice(permits)
print(api)

xwell = ext_fullset[ext_fullset.WELL_PERMIT_NUM == api]['PRODUCTION_PERIOD_START_DATE']
ywell = ext_fullset[ext_fullset.WELL_PERMIT_NUM == api]['AVERAGE_GAS_DAILY_PRODUCTION']
ywell2 = ext_fullset[ext_fullset.WELL_PERMIT_NUM == api]['FORECAST_OUT']

fig = go.Figure()
fig.add_trace(go.Scatter(
                    x=xwell,
                    y=ywell,
                    mode='lines',
                    name='Daily Production'))
fig.add_trace(go.Scatter(
                    x=xwell,
                    y=ywell2,
                    mode='lines',
                    name='Forecast'))

fig.show()


# In[ ]:


#Drop Months past Minimum number of months
ext_fullset.drop(ext_fullset[ext_fullset['Months'] > 90].index, inplace=True)


# In[ ]:


#Export to csv
ext_fullset.reset_index(inplace=True, drop=True)
ext_fullset.to_csv(saved_file)

