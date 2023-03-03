#%% packages
from ast import Mult
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader 
import seaborn as sns
import numpy as np
from collections import Counter
# %% data prep
X, y = make_multilabel_classification(n_samples=10000, n_features=10, n_classes=3, n_labels=2)
X_torch = torch.FloatTensor(X)
y_torch = torch.FloatTensor(y)

# %% train test split
X_train, X_test, y_train, y_test = train_test_split(X_torch, y_torch, test_size = 0.2)


# %% dataset and dataloader

# create instance of dataset

# create train loader


# %% model
# set up model class
# topology: fc1, relu, fc2
# final activation function??


# define input and output dim

# create a model instance

# %% loss function, optimizer, training loop
# set up loss function and optimizer

# implement training loop
    
    
# %% losses
# plot losses

# %% test the model
# predict on test set

#%% Naive classifier accuracy
# convert y_test tensor [1, 1, 0] to list of strings '[1. 1. 0.]'

# get most common class count

