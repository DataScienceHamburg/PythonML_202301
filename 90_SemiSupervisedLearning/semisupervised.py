#%% packages
import pandas as pd
import numpy as np
import random
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
#%%
X, y = fetch_california_housing(return_X_y = True)
# %% train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=42)

# %% scale the data
scaler = StandardScaler()
# scale train
X_train_scaled = scaler.fit_transform(X_train)
# apply scaling from train to test
X_test_scaled = scaler.transform(X_test)

#%%
model_list = [
    LinearRegression(),
    RandomForestRegressor(),
    XGBRegressor()
]
# %%
for m in model_list:
    n_folds = 10
    scores = cross_val_score(estimator=m, X= X_train_scaled, y=y_train, cv = n_folds, scoring = 'r2')
    print(f"{m.__class__.__name__}: R**2 {np.round(scores.mean(), 3)}")
# %%
