# An Ising model approach to financial data analysis

## Project Overview
This repository contains the code used in my Bachelor's Thesis. It primarily implements one of the most famous statistical mechanics models, the Ising model, to study financial data. 
The project explores how to "learn" the model's parameters from the data to further recover the behaviour of the system, namely, its first and second momentum. Additionally, other interesting features 
are explored:
- This is the case for defining a notion of entropy to study the order-disorder phases of the system.
- Minimum Spanning Tree from the inferred parameters.

This study focuses on 24 IBEX 35 stocks, the most important stock index in Spain, and examines whether binary spin approximations can capture essential market behaviors. 

## Objectives
- Model financial markets using the Ising model, treating stock returns as spins.
- Infer interaction networks from stock prices using the inverse Ising model (mean-field method).
- Compare independent vs. pairwise interaction models and evaluate their effectiveness.
- Assess binarization effects and its impact on capturing real market behaviors.
- Investigate the role of highly correlated stocks (e.g., BBVA & Santander) in distorting model predictions via Kullback-Leibler divergence.
