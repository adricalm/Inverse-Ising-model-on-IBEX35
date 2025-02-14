# An Ising model approach to financial data analysis

## Project Overview
This repository contains the basic code used in my Bachelor's Thesis. It primarily implements one of the most famous statistical mechanics models, the Ising model, to study financial data. 
The project focuses on learning the model’s parameters from stock return data to recover key statistical properties of the system, particularly its first and second moments (mean and correlations). 
Additionally, other interesting features are explored:
- Entropy analysis: Investigating the system’s order-disorder transitions defining a notion of entropy.
- Minimum Spanning Tree (MST): Constructing an MST based on the inferred interaction network.

The analysis is applied to 24 stocks from the IBEX 35, Spain’s leading stock index, to assess whether binary spin approximations effectively capture real market behaviors.

## Objectives
- Model financial markets using the Ising model, treating stock returns as spins.
- Infer interaction networks among stocks via the inverse Ising model (mean-field method).
- Compare independent vs. pairwise interaction models and evaluate their effectiveness.
- Assess binarization effects and its impact on capturing real market behaviors.
- Investigate the role of highly correlated stocks (e.g., BBVA & Santander) in distorting model predictions via Kullback-Leibler divergence.
