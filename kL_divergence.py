import pandas as pd
import numpy as np
import os
from datetime import datetime
from collections import Counter
import matplotlib.pyplot as plt
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


all_files_no_bbva_san = files = [ 'ANA.MC 00-24.csv','ACS.MC 02-24.csv', 
         'ACX 00-24.csv', 'BKT 00-24.csv', 
         'CABK.MC 07-24.csv',  'ENG.MC 02-24.csv', 
         'ELE.MC 00-24.csv', 'IBE.MC 00-24.csv', 
          'FER.MC 00-24.csv', 'GRF.MC 07-24.csv', 
          'IDR.MC 00-24.csv', 'ITX.MC 01-24.csv' ,
         'MAP.MC 01-24.csv', 'MEL.MC 00-24!.csv', 
         'MTS.MC 06-24.csv', 'NTGY.MC 00-24.csv', 
         'RED.MC (REE) 00-24.csv', 'REP.MC 00-24.csv', 
          'SAB.MC 00-24.csv', 'SGREN 07-23.csv',
         'TEF.MC 00-24.csv',  'VIS.MC 04-24.csv']


kls_data1 = []
kls_data2 = []

kls_data1_c = []
kls_data2_c = []
# Not compute the KL divergence if the files list was already 
used_permutations = set()
fixed_stocks = ['BBVA 00-24.csv', 'SAN.MC 00-24.csv'] 

loop = 20

n_errors = 0
for i in range(loop):

    while True:
        # Generate a sublist of random elements
        filess = random.sample(all_files_no_bbva_san, 6) #list of 6 random stocks from all stocks
        files = filess + fixed_stocks # Total of 8 stocks with always bbva and san
        # Convert sublist to a tuple to make it hashable
        files_tuple = tuple(files)
        
        # Check if this permutation has already been used
        if  files_tuple  not in used_permutations:
            used_permutations.add(files_tuple)
            break
    
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
    condition_num = np.linalg.cond(C)
    
    if condition_num < 1000:
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
        
        for i in range(cols):
            J[i,i] = 0
            
        Jij = []
        for i in range(cols):
            for j in range(cols):
                Jij.append(J[i,j])
        
        
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
        
        # Partition funciton (normalitzation factor)
        Z = 0
        Hs = []
        for s in s_microstates:
            E = Hamiltonian(s)
            Hs.append(E)
            Z += np.exp(-E)
        
        
        @jit
        def probability(s):
            return (1/Z)*np.exp(-Hamiltonian(s))
        
        
        
        '''6. Kullback-Leibler divergence'''
        # a first row of the prob when spin_i is +1 and a second row when spin_i is -1. 
        # Here a probability is actually the frequency of appearence: nº +1/samples and nº -1/samples
        probabilities_1 = np.zeros((2, cols)) 
        for j in range(cols):
            pj = 0
            pj_ = 0
            for i in range(rows):
                if spins_matrix[i,j] == 1:
                    pj += 1 
                elif spins_matrix[i,j] == -1:
                    pj_ += 1 
            probabilities_1[0,j] = pj/rows
            probabilities_1[1,j] = pj_/rows
                    
                
        @jit
        def probability1(s):
            p = 1
            for i, spin in enumerate(s):
                if spin == 1:
                    p *= probabilities_1[0,i]
                elif spin == -1:
                    p *= probabilities_1[1,i]
                    
            return p
        
        # Preparation to find probabilities of the samples data
        samples_counts = Counter(tuple(row) for row in spins_matrix) #unique samples in data and nº of times appear
        n_samples = len(samples_counts) #nº of unique samples
        samples_prob = [] 
        for sample, count in samples_counts.items():
            samples_prob.append([sample, count/rows])
        
        # Pdata, P1 independent model and P2 pairwise
        Pdata = []
        P1 = []
        P2 = []
        for i in range(n_samples):
            Pdata.append(samples_prob[i][1])
            P1.append(probability1(samples_prob[i][0]))
            P2.append(probability(samples_prob[i][0]))
        
        # Pdata, P1 and P2 have to be prob distributions so sum = 1
        P1 = [p1*(sum(P1)**(-1)) for p1 in P1] #normalize P1
        P2 = [p2*(sum(P2)**(-1)) for p2 in P2] #normalize P2
        
        # Function to compute KL divergence
        def kl_divergence(p_data, p_model):
            return entropy(p_data, p_model)
        
        # Compute KL divergence between Pdata and P1
        kl_div_Pdata1 = kl_divergence(Pdata, P1)
        # Compute KL divergence between Pdata and P2
        kl_div_Pdata2 = kl_divergence(Pdata, P2)
        kl_div_P1P2 = kl_divergence(P1, P2)
        
        kls_data1_c.append(kl_div_Pdata1)
        kls_data2_c.append(kl_div_Pdata2)
    
    else:
        n_errors += 1
           
        
'''CALCULATIONS WITHOUT BBVA AND SAN'''
for i in range(loop-n_errors):

    while True:
        # Generate a sublist of random elements
        filess = random.sample(all_files_no_bbva_san, 8) #list of 6 random stocks from all stocks
        files = filess #+ fixed_stocks # Total of 8 stocks with always bbva and san
        # Convert sublist to a tuple to make it hashable
        files_tuple = tuple(files)
        
        # Check if this permutation has already been used
        if  files_tuple  not in used_permutations:
            used_permutations.add(files_tuple)
            break
    
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
    
    for i in range(cols):
        J[i,i] = 0
        
    Jij = []
    for i in range(cols):
        for j in range(cols):
            Jij.append(J[i,j])
    
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
    
    # Partition funciton (normalitzation factor)
    Z = 0
    Hs = []
    for s in s_microstates:
        E = Hamiltonian(s)
        Hs.append(E)
        Z += np.exp(-E)
    
    
    @jit
    def probability(s):
        return (1/Z)*np.exp(-Hamiltonian(s))
    
    
    
    '''6. Kullback-Leibler divergence'''
    # a first row of the prob when spin_i is +1 and a second row when spin_i is -1. 
    # Here a probability is actually the frequency of appearence: nº +1/samples and nº -1/samples
    probabilities_1 = np.zeros((2, cols)) 
    for j in range(cols):
        pj = 0
        pj_ = 0
        for i in range(rows):
            if spins_matrix[i,j] == 1:
                pj += 1 
            elif spins_matrix[i,j] == -1:
                pj_ += 1 
        probabilities_1[0,j] = pj/rows
        probabilities_1[1,j] = pj_/rows
                
            
    @jit
    def probability1(s):
        p = 1
        for i, spin in enumerate(s):
            if spin == 1:
                p *= probabilities_1[0,i]
            elif spin == -1:
                p *= probabilities_1[1,i]
                
        return p
    
    # Preparation to find probabilities of the samples data
    samples_counts = Counter(tuple(row) for row in spins_matrix) #unique samples in data and nº of times appear
    n_samples = len(samples_counts) #nº of unique samples
    samples_prob = [] #list of lists of the form [sample spins, prob. of appearing]
    for sample, count in samples_counts.items():
        samples_prob.append([sample, count/rows])
    
    # Pdata, P1 independent model and P2 pairwise
    Pdata = []
    P1 = []
    P2 = []
    for i in range(n_samples):
        Pdata.append(samples_prob[i][1])
        P1.append(probability1(samples_prob[i][0]))
        P2.append(probability(samples_prob[i][0]))
    
    # Pdata, P1 and P2 have to be prob distributions so sum = 1
    P1 = [p1*(sum(P1)**(-1)) for p1 in P1] #normalize P1
    P2 = [p2*(sum(P2)**(-1)) for p2 in P2] #normalize P2
    
    # Function to compute KL divergence
    def kl_divergence(p_data, p_model):
        return entropy(p_data, p_model)
    
    # Compute KL divergence between Pdata and P1
    kl_div_Pdata1 = kl_divergence(Pdata, P1)
    # Compute KL divergence between Pdata and P2
    kl_div_Pdata2 = kl_divergence(Pdata, P2)
    kl_div_P1P2 = kl_divergence(P1, P2)
    
    kls_data1.append(kl_div_Pdata1)
    kls_data2.append(kl_div_Pdata2)
    
    
print('number of errors', n_errors, 'of a total tries of', loop)
end_time = time.time()

execution_time = end_time - start_time

print('Time to execute:', execution_time)

print('Mitjana data-1', np.mean(kls_data1))
print('Mitjana data-2', np.mean(kls_data2))
print('Std data-1', np.std(kls_data1))
print('Std data-2', np.std(kls_data2))

Deltas = []
for i in range(len(kls_data1)):
    Deltas.append(kls_data2[i]/kls_data1[i])
    
Delta = np.mean(Deltas)
Delta_std = np.std(Deltas)
print('Delta (mean_deltas)', Delta)
print('std deltas', Delta_std)

print('\n')
print('Mitjana data-1 c', np.mean(kls_data1_c))
print('Mitjana data-2 c', np.mean(kls_data2_c))
print('Std data-1 c', np.std(kls_data1_c))
print('Std data-2 c', np.std(kls_data2_c))

Deltas_c = []
for i in range(len(kls_data1_c)):
    Deltas_c.append(kls_data2_c[i]/kls_data1_c[i])
    
Delta_c = np.mean(Deltas_c)
Delta_std_c = np.std(Deltas_c)
print('Delta c(mean_deltas)', Delta_c)
print('std deltas c', Delta_std_c)


# Configure Matplotlib to use LaTeX for text rendering
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 20
})

print(len(kls_data1))
print(len(kls_data2))
plt.figure(figsize=(10, 6))
# Plot the histogram
plt.hist(kls_data1, bins = 5, density=True, edgecolor='black', color='grey', alpha=0.6)
# Add labels and title
plt.xlabel(r'$D_{KL}(p_{data} ||  p_{ind})$', fontsize=22)
#plt.savefig('hist_KL_data_ind.pdf', dpi=300, bbox_inches='tight')
# Show the plot
# Plot the histogram
plt.hist(kls_data1_c, bins = 10, density=True, edgecolor='black', color='grey', alpha=0.6)
# Add labels and title
plt.xlabel(r'$D_{KL}(p_{data} ||  p_{ind_c})$', fontsize=22)
#plt.savefig('hist_KL_data_ind_c.pdf', dpi=300, bbox_inches='tight')
# Show the plot
plt.show()


plt.figure(figsize=(10, 6))
# Plot the histogram
plt.hist(kls_data2, bins = 5, density=True, edgecolor='black', color='grey', alpha=0.6)
plt.hist(kls_data2_c, bins = 10, density=True, edgecolor='black', color='red', alpha=0.6)
# Add labels and title
plt.xlabel(r'$D_{KL}(p_{data} ||  p_{Ising})$', fontsize=22)
#plt.savefig('hist_KL_data_Ising_c.pdf', dpi=300, bbox_inches='tight')
# Show the plot
plt.show()


