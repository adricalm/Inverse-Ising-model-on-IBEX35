import pandas as pd
import numpy as np
import os
from datetime import datetime
from collections import Counter
import matplotlib.pyplot as plt
from numba import jit
from scipy import stats


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

df_stocks = [] #list of dataframes of each stock sliced all with the same date values

# Matching trading days all stocks
trading_days = {key: None for key in files} 
for f in files:
    file_path = os.path.join(d_path, f)

    df = pd.read_csv(file_path)

    index_first_day = df[df['Date'] == first_day].index[0]
    index_last_day = df[df['Date'] == last_day].index[0]

    days = df.loc[index_first_day:index_last_day, 'Date']
    trading_days[f] = days

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
close_stocks = [] #DataFrame of price stocks
dates = []
count = 0
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
    
    close_stocks.append(df['Close'])
  
    if count == 0:
        dates.append([df['Date']])
    count += 1

total_trading_days_str = dates[0][0].values
total_trading_days = [datetime.strptime(date, '%d/%m/%Y') for date in total_trading_days_str ]


'''2. START COMPUTING'''
# 1.Spins matrix
spins_matrix = np.array(spins_matrix).T
spins_matrix_shape = spins_matrix.shape

rows = spins_matrix_shape[0] #number of samples/tradying days (rows) in the matrix
cols = spins_matrix_shape[1] #number of columns (stocks) in the matrix
print('trading days (rows)', rows)
print(first_day, last_day)
# 2.Magnetizations from data average
m_D = np.zeros(cols) 
for i in range(cols):
    m_D[i] = np.mean(spins_matrix[:,i])

stocks = [i for i in range(cols)]

# 3.Covariance matrix
#C_ij = <si*sj> - mi*mj
si_sj_D = np.zeros((cols, cols))
C = np.zeros((cols, cols))
for i in range(cols):
    for j in range(cols):
        vector_i = np.array(spins_matrix[:,i])
        vector_j = np.array(spins_matrix[:,j])
        si_sj_D[i,j] = np.mean(vector_i*vector_j)
        C[i,j] = si_sj_D[i,j] - m_D[i]*m_D[j]


'''3. MEAN-FIELD APPROACH'''
#with this method, couplings matrix J_{i,j} == C_inv_{i,j} only if i != j, so the diagonal J_{i,i} cannot be found
print('determinant', np.linalg.det(C))
print('condition number', np.linalg.cond(C))
A = np.zeros((cols, cols))
for i in range(cols):
    A[i,i] = 1 - m_D[i]**2

J = np.linalg.inv(A) - np.linalg.inv(C)

h = np.zeros(cols)
for i in range(cols):
    h_i = 0
    h_i += np.arctanh(m_D[i])
    for j in range(cols):
        h_i += -J[i,j]*m_D[j] 

    h[i] = h_i


'''PARTITION FUNCTION AND PROBABILITY'''
# Generate permutations (all possible microstates)
def generate_permutations(n_stocks):
    num_permutations = 2**n_stocks  # Total number of permutations
    permutations = np.zeros((num_permutations, n_stocks), dtype=int)
    
    for i in range(num_permutations):
        # Directly fill the permutation matrix without intermediate string conversion
        for bit in range(n_stocks):
            permutations[i, n_stocks-1-bit] = -1 if (i >> bit) & 1 == 0 else 1
            
    return permutations


# Hamiltonian
@jit
def Hamiltonian(s):
    H = np.zeros((1,1),  dtype=np.float64)
    for i in range(cols):
        H += -h[i]*s[i]
        for j in range(cols):
            if i != j:
                H += -0.5*J[i,j]*s[i]*s[j] 
    return H[0,0]


s_microstates = generate_permutations(cols) #microstates matrix
n_microstates = s_microstates.shape[0]
# Partition funciton (normalitzation factor)
Z = 0
Hs = []
for s in s_microstates:
    E = Hamiltonian(s)
    Hs.append(E)
    Z += np.exp(-E)
print('Partition function: ', Z)

@jit
def probability(s):
    return (1/Z)*np.exp(-Hamiltonian(s))


'''5. STUDYING INTERESTING SHIT'''

'''5.0. Computing entropy with MF Entropy'''
S_D = [] #we compute the entropy on periods to see the evolution
d0 = 0
dn = 300 #nº of days (samples) we take to compute entorpy

original_dn = dn
for i in range(rows-dn): #we don't compute the last period
    S = 0
    for j in range(cols):
        m_j = np.mean(spins_matrix[d0:dn,j])
        if m_j > 1: print(m_j)
        S += ((1+m_j)/2)*np.log((1+m_j)/2) + ((1-m_j)/2)*np.log((1-m_j)/2)
        S = -S
    S_D.append(S)
    
    #update period window
    d0 += 1
    dn += 1
dn = original_dn


abs_S_D = []
for i in range(len(S_D)):
    abs_S_D.append(abs(S_D[i]))
    
norm_S_D = S_D/max(abs_S_D )

'''5.1 Visualize the evolution of the normalized sum of all stocks'''
#normalizing stocks
normalized_stocks = []
for i in range(cols):
    normalized_stock = close_stocks[i]/max(close_stocks[i])
    normalized_stocks.append(normalized_stock)
normalized_stocks = np.array(normalized_stocks).T #rows = trading days, cols = stocks

mean_stock = np.zeros(rows)
for i in range(rows-dn):
    value_i = 0
    for j in range(cols):
        value_i += normalized_stocks[i,j]
    mean_stock[i] = value_i/cols

mean_stock = mean_stock[:-dn]

norm_S_D = norm_S_D[:-dn]
total_days = rows-dn
days_stocks = [i for i in range(total_days)]
days_S_D = [i for i in range(dn, total_days)]

clean_S_D = []
clean_stocks = []
clean_days_stocks = []
clean_days_S_D = []
clean_trading_days = []
for i in range(len(days_stocks)):
    if i%10 == 0:
        clean_stocks.append((mean_stock[i]-mean_stock[0])*1.5)
        clean_days_stocks.append(i)
        clean_trading_days.append(total_trading_days[i])
        
for i in range(len(days_S_D)):
    if i%10 == 0:
        clean_S_D.append(norm_S_D[i]*0.7)
        clean_days_S_D.append(i+dn)
    
a = int(dn/10)
clean_days_S_D = clean_trading_days[a:]

plt.figure(figsize=(12, 6))
plt.plot(clean_trading_days, clean_stocks, linewidth = 2)
plt.plot(clean_days_S_D, clean_S_D, '--', color = 'red', linewidth = 2, )
plt.show()