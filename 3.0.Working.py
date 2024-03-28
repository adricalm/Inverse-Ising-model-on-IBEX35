import pandas as pd
import numpy as np
import os
from datetime import datetime
from collections import Counter
import matplotlib.pyplot as plt
from numba import jit
from scipy import stats

''' NOTES '''
#I remove 'GAMQ 10-24.csv', 'ICAG (IAG) 12-24.csv' because of its short trading time
#and others are removed also at 1.0.Start_computing.py because because they have around 200 less trading days 
#than the rest, I guess because where taken from investing which provides less days idk.

d_path = '/Users/adria/Desktop/uni/4t/TFG/Inverse Ising Model TFG/IBEX 35 data/cotizacion_acciones'
files = ['ABE.MC 04-18.csv',  'BBVA 00-24.csv', 'SGREN 07-23.csv', 'TL5 04-23.csv',
          'BKT.MC 00-24.csv','ELE.MC 00-24.csv','FCC.MC 00-24.csv','IBE.MC 00-24.csv','MAP.MC 01-24.csv', 'NTGY.MC 00-24.csv', 
        'RED.MC (REE) 00-24.csv', 'REP.MC 00-24.csv', 'SAN.MC 00-24.csv', 'VIS.MC 04-24.csv']

'''1. DATA PREPARATION'''
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

# Saving trading days of each stocks (because I want to eliminate the days, maybe because of different data sources or things of the stock itself, some stocks I have data and others don't)
trading_days = {key: None for key in files} #create an empty dictionary with the trading days of each stock
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

# Identify values not in all keys using a for loop for better understanding
for value, count in value_counts.items():
    if count < len(trading_days):
        days_not_in_all_stocks.append(value)


spins_matrix = []
for df in df_stocks:
    
    # Ensuring no null values 
    #print(df.isnull().sum(), '\n')

    # Looking which types of data in column
    Open_datatypes = df['Open'].apply(type)    
    types = []
    for i in Open_datatypes:
        if i in types:
            pass
        else:
            types.append(i)
    #print(types)
    
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


stocks = [i for i in range(cols)]
plt.scatter(stocks, m_D)
plt.xlabel('Stocks (spins)')
plt.ylabel('Magnetization from data')
plt.show()

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


'''3. MEAN-FIELD APPROACH'''
#with this method, couplings matrix J_{i,j} == C_inv_{i,j} only if i != j, so the diagonal J_{i,i} cannot be found
J = -np.linalg.inv(C)

h = np.zeros(cols)
for i in range(cols):
    h_i = 0
    h_i += np.arctanh(m_D[i])
    for j in range(cols):
        if i != j:
            h_i -= J[i,j]*m_D[j]

    h[i] = h_i

plt.hist(h)
plt.title('h vector')
plt.show()
Jij = []
for i in range(cols):
    for j in range(cols):
        Jij.append(J[i,j])
plt.hist(Jij)
plt.title('J matrix')
plt.show()


#J graphically
plt.imshow(J, cmap='bwr', interpolation='nearest', vmin=-1, vmax=1)
# Overlay the diagonal elements with gray
for i in range(len(J)):
    plt.gca().add_patch(plt.Rectangle((i-0.5, i-0.5), 1, 1, color='gray', alpha=1))
# Add a colorbar to show the mapping from color intensity to matrix values
plt.colorbar()
plt.title('Matrix Visualization')
plt.show()


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

# Show 'Energy' of the configuration in the 32 configuration
n = 32
print(f'H with sample {n}: ', Hamiltonian(spins_matrix[n])) 


Hs_D = []
for i in range(rows):
    Hs_D.append(Hamiltonian(spins_matrix[i]))

indices = [i for i in range(rows)]
plt.plot(indices, Hs_D)
plt.show()

plt.hist(Hs_D, bins=30, density=False)
plt.xlabel('Energy data')
plt.show()

s_microstates = generate_permutations(cols) #microstates matrix

# Partition funciton (normalitzation factor)
Z = 0
Hs = []
for s in s_microstates:
    E = Hamiltonian(s)
    Hs.append(E)
    Z += np.exp(-E)


plt.hist(Hs, bins=35, density = False)
plt.xlabel('Energy from pdf')
plt.show()

print('Partition function: ', Z)

@jit
def probability(s):
    return (1/Z)*np.exp(-Hamiltonian(s))



'''COMPUTING OBSERVABLES (MAGNETIZATIONS AND CORRELATIONS) WITH PDF'''

#Magnetizations from pdf
m = np.zeros(cols) #magnetizations from boltzmann distribution


for j in range(cols):
    for i in range(s_microstates.shape[0]):
        m[j] += s_microstates[i,j]*probability(s_microstates[i,:])

# Comparaison of m_D and m (m_pdf)
slope, intercept, r_value, p_value, std_err = stats.linregress(m_D, m)
print('Coefficient of determination:', r_value**2)


print('m_D =', m_D)
print('m_pdf =',m)
equality = np.linspace(min(m_D), max(m_D), 100)
plt.scatter(m_D, m)
plt.xlabel('m_data')
plt.ylabel('m_pdf')
for x, y, name in zip(m_D, m, files):
    plt.scatter(x, y, label=name)

# Add legend to the top

plt.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=2)
plt.plot(equality, equality)
plt.title(f'$R^2 = $ {r_value**2}')
plt.show()


# 'Correlations' matrix from pdf, i.e., second order moment
si_sj = np.zeros((cols, cols)) # <si*sj> from boltzmann

for i in range(cols):
    for j in range(cols):
            if i != j:
                for k in range(s_microstates.shape[0]):
                    si_sj[i,j] += s_microstates[k,i]*s_microstates[k,j]*probability(s_microstates[k,:])

                
# Comparaison of data and pdf

si_sj_D_list = [] #all <si*sj> values except for i = j
for i in range(cols):
    for j in range(cols):
        if i != j:
            si_sj_D_list.append(si_sj_D[i,j])
            
            
si_sj_list = []
for i in range(cols):
    for j in range(cols):
        if i != j:
            si_sj_list.append(si_sj[i,j])
            
 
    
slope, intercept, r_value, p_value, std_err = stats.linregress(si_sj_D_list, si_sj_list)
print('Coefficient of determination:', r_value**2)


print('si_sj_D=', si_sj_D_list )
print('si_sj_pdf =', si_sj_list)
equality = np.linspace(min(si_sj_D_list), max(si_sj_D_list), 100)
plt.scatter(si_sj_D_list, si_sj_list)
plt.xlabel('si_sj_D')
plt.ylabel('si_sj_pdf')
plt.plot(equality, equality)
plt.title(f'$R^2 = $ {r_value**2}')
plt.show()























