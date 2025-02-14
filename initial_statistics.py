# -*- coding: utf-8 -*-
"""
Created on Sat May 25 19:30:45 2024

@author: adria
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
from collections import Counter
from scipy.stats import skew, kurtosis

d_path = 'stocks_data'
files = [ 'ANA.MC 00-24.csv','ACS.MC 02-24.csv', 
         'ACX 00-24.csv', 
         'BBVA 00-24.csv', 'BKT 00-24.csv', 'CABK.MC 07-24.csv',  
         'ENG.MC 02-24.csv', 'ELE.MC 00-24.csv', 
          'FER.MC 00-24.csv', 'GRF.MC 07-24.csv', 
         'IBE.MC 00-24.csv', 
          'IDR.MC 00-24.csv', 'ITX.MC 01-24.csv' ,
         'MAP.MC 01-24.csv', 'MEL.MC 00-24!.csv', 
         'MTS.MC 06-24.csv', 'NTGY.MC 00-24.csv', 
         'RED.MC (REE) 00-24.csv', 'REP.MC 00-24.csv', 'SAN.MC 00-24.csv' ,
          'SAB.MC 00-24.csv', 'SGREN 07-23.csv',
         'TEF.MC 00-24.csv',  'VIS.MC 04-24.csv']

# Findint the latest starting date, the earliest delisting date 
start_dates = [] #first date of each stock
last_dates = [] #last date trading
for f in files:
    file_path = os.path.join(d_path, f)
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)

    start_dates.append(df['Date'].iloc[0])
    last_dates.append(df['Date'].iloc[-1])

dates = [datetime.strptime(date, '%d/%m/%Y') for date in start_dates]
first_day = max(dates) #last stock to start trading: Day 0 of data
first_day = first_day.strftime('%d/%m/%Y')

dates = [datetime.strptime(date, '%d/%m/%Y') for date in last_dates]
last_day = min(dates) #first stock to stop trading: Last Day of data
last_day = last_day.strftime('%d/%m/%Y')

df_stocks = []

# Saving trading days of each stocks 
trading_days = {key: None for key in files}
for f in files:
    file_path = os.path.join(d_path, f)

    df = pd.read_csv(file_path)

    index_first_day = df[df['Date'] == first_day].index[0]
    index_last_day = df[df['Date'] == last_day].index[0]
    #Save all traded days in the dict trading_days
    days = df.loc[index_first_day:index_last_day, 'Date']
    trading_days[f] = days

    # Append Data Frame of each stock
    df = df.loc[index_first_day:]
    df = df.loc[:index_last_day].reset_index(drop=True)
    df_stocks.append(df)

value_counts = Counter()
# Count occurrences of each value
for values in trading_days.values():
    value_counts.update(values)

#Initialize an empty list to store values not in all keys
days_not_in_all_stocks = []

# Identify values not in all keys using a for loop for better understanding
for value, count in value_counts.items():
    if count < len(trading_days):
        days_not_in_all_stocks.append(value)

close_stocks = [] 
dates = []
count = 0
spins_matrix = []
returns_matrix = []
for df in df_stocks:
    # Looking which types of data in column
    Open_datatypes = df['Open'].apply(type)    
    types = []
    for i in Open_datatypes:
        if i in types:
            pass
        else:
            types.append(i)
    
    df = df[~df['Date'].isin(days_not_in_all_stocks)].reset_index(drop=True)
    
    # Convert str values from investing.com files to float
    if str in types:
        df['Open'] = df['Open'].str.replace(',', '').astype(float)
        df['Close'] = df['Close'].str.replace(',', '').astype(float)
    
    # 'Binarize' data
    returns = np.diff(df['Close'], axis=0)
    returns_matrix.append(returns)
    spins = np.zeros(len(returns))
    for i in range(len(returns)):
        if returns[i] >= 0.0:
            spins[i] = 1
        else:
            spins[i] = -1
            
    
    spins_matrix.append(spins)
    
    close_stocks.append(df['Close'].iloc[:-1])
    
    if count == 0:
        dates.append([df['Date'].iloc[:-1]])
    count += 1


total_trading_days_str = dates[0][0].values
total_trading_days = [datetime.strptime(date, '%d/%m/%Y') for date in total_trading_days_str]


#5.1 Visualize the evolution of the normalized sum of all stocks#because stocks have different tradying price, we normalize all of them 
#with its high equal 1.

cols = len(files)
rows = len(total_trading_days)

normalized_stocks = []
for i in range(cols):
    normalized_stock = close_stocks[i]/max(close_stocks[i])
    normalized_stocks.append(normalized_stock)
normalized_stocks = np.array(normalized_stocks).T #rows = trading days, cols = stocks

mean_stock = np.zeros(rows)
for i in range(rows):
    value_i = 0
    for j in range(cols):
        value_i += normalized_stocks[i,j]
    mean_stock[i] = value_i/cols


total_days = rows
days_stocks = [i for i in range(total_days)]


window = 250
a = 0
b = window
print('Temporal window:', window)
# 1.1 Log returns (stocks returns)
returns_matrix = np.array(returns_matrix).T 
spins_matrix = np.array(spins_matrix).T
prices_matrix = np.matrix(close_stocks).T


Means = []
Varss = []
Skes = []
Kurts = []
Means_b = []
Varss_b = []
Skes_b = []
Kurts_b = []
for i in range(rows):
    # Find the raw matrix and the binarized matrix
    prices_matrix_i = prices_matrix[a:b,:]
    s_raw = np.diff(prices_matrix_i, axis=0) / prices_matrix_i[:-1, :]

    spins_matrix_i = spins_matrix[a:b,:]
    s_bin = spins_matrix_i 
    
    # Initialize arrays to store moments for each stock
    means = np.zeros(s_raw.shape[1])
    variances = np.zeros(s_raw.shape[1])
    skewnesses = np.zeros(s_raw.shape[1])
    kurtoses = np.zeros(s_raw.shape[1])
    
    means_b = np.zeros(s_raw.shape[1])
    variances_b = np.zeros(s_raw.shape[1])
    skewnesses_b = np.zeros(s_raw.shape[1])
    kurtoses_b = np.zeros(s_raw.shape[1])
    
    # Step 3: Compute moments for each stock
    for i in range(s_raw.shape[1]):
        # s raw
        means[i] = np.mean(s_raw[:, i])
        variances[i] = np.std(s_raw[:, i])
        skewnesses[i] = skew(s_raw[:, i])
        kurtoses[i] = kurtosis(s_raw[:, i], fisher=False)
    
        # s bin
        means_b[i] = np.mean(s_bin[:, i])
        variances_b[i] = np.std(s_bin[:, i])
        skewnesses_b[i] = skew(s_bin[:, i])
        kurtoses_b[i] = kurtosis(s_bin[:, i], fisher=False)
        
    #append the mean of the first 4 moments of all stocks
    Means.append(np.mean(means))
    Varss.append(np.mean(variances))
    Skes.append(np.mean(skewnesses))
    Kurts.append(np.mean(kurtoses))
        
    Means_b.append(np.mean(means_b))
    Varss_b.append(np.mean(variances_b))
    Skes_b.append(np.mean(skewnesses_b))
    Kurts_b.append(np.mean(kurtoses_b))
    
    a += 1
    b += 1

plt.figure(figsize=(9, 4))
plt.plot(total_trading_days[:rows-window], Means[:rows-window], label='s raw', color='blue')
plt.plot(total_trading_days[:rows-window], Means_b[:rows-window], label='s bin', color='red')
plt.ylabel('Mean')
plt.legend()
plt.show()

plt.figure(figsize=(9, 4))
plt.plot(total_trading_days[:rows-window], Varss[:rows-window], label='s raw', color='blue')
plt.plot(total_trading_days[:rows-window], Varss_b[:rows-window], label='s bin', color='red')
plt.ylabel('Std')
plt.legend()
plt.show()

plt.figure(figsize=(9, 4))
plt.plot(total_trading_days[:rows-window], Skes[:rows-window], label='s raw', color='blue')
plt.plot(total_trading_days[:rows-window], Skes_b[:rows-window], label='s bin', color='red')
plt.ylabel('Skewness')
plt.legend()
plt.show()

plt.figure(figsize=(9, 4))
plt.plot(total_trading_days[:rows-window], Kurts[:rows-window], label='s raw', color='blue')
plt.plot(total_trading_days[:rows-window], Kurts_b[:rows-window], label='s bin', color='red')
plt.title('Kurts log')
plt.legend()
plt.show()



print('Correlation mean:', np.corrcoef(Means[:rows-window], Means_b[:rows-window]))
print('Correlation variance:', np.corrcoef(Varss[:rows-window], Varss_b[:rows-window]))
print('Correlation skewness:', np.corrcoef(Skes[:rows-window], Skes_b[:rows-window]))
print('Correlation kurtosis:', np.corrcoef( Kurts[:rows-window], Kurts_b[:rows-window]))





