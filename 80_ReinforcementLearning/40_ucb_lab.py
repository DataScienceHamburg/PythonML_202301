#%% packages
import pandas as pd
import numpy as np
from plotnine import ggplot, aes, geom_violin, geom_point
import math
# %% bandit returns
# The designer of the bandits inherently designed average and variation parameters into the machines.
A_mean = .97
B_mean = .98
C_mean = 1.1
D_mean = .99
E_mean = .96
A_sd = 0.05
B_sd = 0.15
C_sd = 0.10
D_sd = 0.11
E_sd = 0.08

bandit_names = ["A", "B", "C", "D", "E"]
bandit_design = pd.DataFrame({'Bandit': bandit_names, 'Mean': [A_mean, B_mean, C_mean, D_mean, E_mean], 'SD': [A_sd, B_sd, C_sd, D_sd, E_sd]})

n_pts =1000

bandits = pd.DataFrame({'A': np.random.normal(A_mean, A_sd, n_pts), 'B': np.random.normal(B_mean, B_sd, n_pts), 'C': np.random.normal(C_mean, C_sd, n_pts), 'D': np.random.normal(D_mean, D_sd, n_pts), 'E': np.random.normal(E_mean, E_sd, n_pts)})
# %%
bandits_long = bandits.melt()

ggplot(data = bandits_long) + aes(x='variable', y='value', fill='variable') + geom_violin()

# %% Exploration / Exploitation with Time
# This information is only know to the designer, but not to the player. The player might perform a set of runs, e.g. 1000 runs, per machine, and find out that machine B is having the highest return. But this strategy is very costly.

# We decide to play 100 rounds. In each round we decide to play on which machine based on previous knowledge.

df_round = pd.DataFrame({'Bandit': bandit_names})
df_round['N'] = 1  # number of times the slot machine was played 
df_round['R'] = 1  # sum of rewards
df_round['r'] = 1  # average rewards
df_round['delta'] = 1E4  # confidence interval
df_round['upper_conf_bound'] = 1E10  # r + delta

max_it = 10000  # maximum iteration

df_iterations = pd.DataFrame({'n': range(1,max_it)})
df_iterations['SelectedBandit'] = np.nan
df_iterations['UCB'] = np.nan
df_iterations['r'] = np.nan
df_iterations['delta'] = np.nan


# %% iterate for each round
for n in range(2, max_it+1):
    if (n==2):
        bandit_chosen_name = 'A'
        bandit_chosen_nr = np.where(df_round['Bandit'] == bandit_chosen_name)[0][0]
    
    # increase nr of runs per selected machine
    df_round.loc[bandit_chosen_nr, 'N'] = df_round.loc[bandit_chosen_nr, 'N'] + 1
    
    # get mean/sd of selected machine
    current_bandit_mean = bandit_design.loc[bandit_chosen_nr, 'Mean']
    current_bandit_sd = bandit_design.loc[bandit_chosen_nr, 'SD']
    # calculate current return
    current_return = np.random.binomial(n=1, p = current_bandit_mean/2) * 2

    # calculate sum of returns
    df_round.loc[bandit_chosen_nr, 'R'] = df_round.loc[bandit_chosen_nr, 'R'] + current_return
    
    # calculate average returns
    df_round.loc[bandit_chosen_nr, 'r'] = df_round.loc[bandit_chosen_nr, 'R'] / df_round.loc[bandit_chosen_nr, 'N']
    
    # calculate confidence interval
    df_round.loc[bandit_chosen_nr, 'delta'] = np.sqrt(2*math.log(n, math.e) / df_round.loc[bandit_chosen_nr, 'N'])
    
    # calculate upper confidence bound
    df_round.loc[bandit_chosen_nr, 'upper_conf_bound'] = df_round.loc[bandit_chosen_nr, 'r'] + df_round.loc[bandit_chosen_nr, 'delta']

    # store selection in df_iteration
    df_iterations.loc[n, 'SelectedBandit'] = bandit_chosen_name
    df_iterations.loc[n, 'UCB'] = df_round.loc[bandit_chosen_nr, 'upper_conf_bound']
    df_iterations.loc[n, 'r'] = df_round.loc[bandit_chosen_nr, 'r']
    df_iterations.loc[n, 'delta'] <- df_round.loc[bandit_chosen_nr, 'delta']

    # define bandit for next run
    max_value = max(df_round['upper_conf_bound'])
    bandit_chosen_nr = df_round['upper_conf_bound'].tolist().index(max_value)
    bandit_chosen_name = bandit_design.loc[bandit_chosen_nr, 'Bandit']

df_round
# %% visualise progress over time
ggplot(data = df_iterations[['n', 'SelectedBandit', 'r']].dropna()) + aes(x='n', y='r', color='SelectedBandit') + geom_point()

# %%
