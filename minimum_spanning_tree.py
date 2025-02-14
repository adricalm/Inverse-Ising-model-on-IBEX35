# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 15:31:04 2024

@author: adria
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.sparse.csgraph import minimum_spanning_tree
import os
from datetime import datetime
from collections import Counter
from numba import jit
from scipy import stats
from scipy.stats import entropy
import random
import time


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

# Configure Matplotlib to use LaTeX for text rendering
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 20
})

start_time = time.time()
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

    # Read the CSV file into a pandas DataFrame
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
for value, count in value_counts.items():
    if count < len(trading_days):
        days_not_in_all_stocks.append(value)


spins_matrix = []
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
    difference = df['Close'] - df['Open']

    for i in range(len(difference)):
        if difference[i] >= 0.0:
            difference[i] = 1
        else:
            difference[i] = -1

    spins = difference
    spins_matrix.append(spins.values)
  

'''2. START COMPUTING'''
# 1.Spins matrix
spins_matrix = np.array(spins_matrix).T
spins_matrix_shape = spins_matrix.shape


rows = spins_matrix_shape[0] #number of samples/tradying days (rows) in the matrix
cols = spins_matrix_shape[1] #number of columns (stocks) in the matrix


# 2.Magnetizations from data average
m_D = np.zeros(cols) 
for i in range(cols):
    m_D[i] = np.mean(spins_matrix[:,i])


# 3.Matrix covariance
#C_ij = <si*sj> - mi*mj
si_sj_D = np.zeros((cols, cols))
C = np.zeros((cols, cols))
for i in range(cols):
    for j in range(cols):
        vector_i = np.array(spins_matrix[:,i])
        vector_j = np.array(spins_matrix[:,j])
        si_sj_D[i,j] = np.mean(vector_i*vector_j)
        C[i,j] = si_sj_D[i,j] - m_D[i]*m_D[j]


'''3.Naive MEAN-FIELD APPROACH'''
#with this method, couplings matrix J_{i,j} == C_inv_{i,j} only if i != j, so the diagonal J_{i,i} cannot be found

A = np.zeros((cols, cols))
for i in range(cols):
    A[i,i] = 1 - m_D[i]**2

J = np.linalg.inv(A) - np.linalg.inv(C)

for i in range(cols):
    J[i,i] = 0



'''MST'''
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.sparse.csgraph import minimum_spanning_tree

# Define the industry classification based on the table provided
industry_classification = {
    'ACS': 'Industrials',
    'ACX': 'Materials',
    'ANA': 'Industrials',
    'BBVA': 'Financials',
    'BKT': 'Financials',
    'CABK': 'Financials',
    'ELE': 'Energy',
    'ENG': 'Energy',
    'FER': 'Industrials',
    'GRF': 'Healthcare',
    'IBE': 'Energy',
    'IDR': 'Information Technology',
    'ITX': 'Consumer Discretionary',
    'MAP': 'Financials',
    'MEL': 'Consumer Discretionary',
    'MTS': 'Materials',
    'NTGY': 'Energy',
    'RED': 'Energy',
    'REP': 'Energy',
    'SAN': 'Financials',
    'SAB': 'Financials',
    'SGRE': 'Industrials',
    'TEF': 'Communication Services',
    'VIS': 'Industrials'
}

# Assume J is the correlation matrix defined before
correlation_matrix = pd.DataFrame(J, columns=list(industry_classification.keys()), index=list(industry_classification.keys()))
# Convert correlation to distance
print(correlation_matrix)
#correlation_matrix.to_csv('J df.csv', index=True)
ones = np.ones((len(files), len(files)))
distance_matrix = ones - abs(correlation_matrix)
np.fill_diagonal(distance_matrix.values, 0)
print(distance_matrix)

print(distance_matrix['SAN'].iloc[3])
# Create a graph from the distance matrix
G = nx.Graph()

# Add nodes
for stock in correlation_matrix.columns:
    G.add_node(stock, industry=industry_classification[stock])

# Add edges
for i in range(len(distance_matrix.columns)):
    for j in range(i + 1, len(distance_matrix.columns)):
        stock_i = distance_matrix.columns[i]
        stock_j = distance_matrix.columns[j]
        G.add_edge(stock_i, stock_j, weight=distance_matrix.iloc[i, j])

# Compute the minimum spanning tree (MST)
mst = nx.minimum_spanning_tree(G)

# Create a color mapping for industries
unique_industries = list(set(industry_classification.values()))
industry_colors = {industry: color for industry, color in zip(unique_industries, plt.cm.tab20.colors)}
node_colors = [industry_colors[G.nodes[n]['industry']] for n in mst.nodes]

# Visualize the MST
pos = nx.spring_layout(mst, seed=42)  # for consistent layout

plt.figure(figsize=(16*0.8, 12*0.8))
nx.draw_networkx(mst, pos, node_color=node_colors, with_labels=True, node_size=800, font_size=10, edge_color='grey', font_color='black')

# Create a legend
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=industry_colors[industry], markersize=10) for industry in unique_industries]
plt.legend(handles, unique_industries, loc='upper left', bbox_to_anchor=(1, 1), title='Sectors')

plt.title('Minimum Spanning Tree of Stock Correlations')
plt.show()
