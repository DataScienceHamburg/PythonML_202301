#%% packages
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import seaborn as sns

#%% data import
cars_file = 'https://gist.githubusercontent.com/noamross/e5d3e859aa0c794be10b/raw/b999fb4425b54c63cab088c0ce2c0d6ce961a563/cars.csv'
cars = pd.read_csv(cars_file)
cars.head()
# %% visualise the data
sns.scatterplot(data=cars, x='wt', y='mpg')
sns.regplot(data=cars, x='wt', y='mpg')
# %% convert data to tensor
X_np = cars.wt.values.reshape(-1, 1)
y_np = cars.mpg.values.reshape(-1, 1)
X = torch.from_numpy(X_np)
y = torch.from_numpy(y_np)

# %% training parameter
NUM_EPOCHS = 1000
LR = 0.001

#%% set up weights and biases
w = torch.rand(1, requires_grad=True, dtype=torch.float64)
b = torch.rand(1, requires_grad=True, dtype=torch.float64)


# %%
for epoch in range(NUM_EPOCHS):
    for i in range(len(X)):
        # forward pass
        y_pred = X[i] * w + b
        
        # calculate loss
        loss_tensor = torch.pow(y_pred - y[i], 2)
        
        # backward pass
        loss_tensor.backward()
        
        # update weights and biases
        with torch.no_grad():
            w -= w.grad * LR
            b -= b.grad * LR
            w.grad.zero_()
            b.grad.zero_()
        print(loss_tensor.data[0])
# %% check results
w.item()

# %%
y_pred = (X*w +b).detach().numpy()

# %% visualise results
sns.scatterplot(x=X_np[:, 0], y = y_np[:, 0])

# %%
sns.scatterplot(x=X_np[:, 0], y = y_pred[:, 0])
# %%
