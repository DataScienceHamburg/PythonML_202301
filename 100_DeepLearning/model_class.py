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

#%% visualise the model
sns.scatterplot(x='wt', y='mpg', data=cars)
sns.regplot(x='wt', y='mpg', data=cars)

#%% convert data to tensor
X_list = cars.wt.values
X_np = np.array(X_list, dtype=np.float32).reshape(-1,1)
y_list = cars.mpg.values
y_np = np.array(y_list, dtype=np.float32).reshape(-1,1)
X = torch.from_numpy(X_np)
y_true = torch.from_numpy(y_np)

#%% model class
class LinearRegressionTorch(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegressionTorch, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        return self.linear(x)

#%%    
input_dim = X_np.shape[1]
output_dim = y_np.shape[1]

model = LinearRegressionTorch(input_size=input_dim, output_size = output_dim)
        
# %% Loss Function
loss_fun = nn.MSELoss()

#%% Optimizer
LR = 0.008
optimizer = torch.optim.SGD(model.parameters(), lr=LR)

# %% perform training
NUM_EPOCHS = 5000
losses, slope, bias = [], [], []

for epoch in range(NUM_EPOCHS):
    
    # set grads to zero
    optimizer.zero_grad()
    
    # forward pass
    y_pred = model(X)
    
    # calc loss
    loss = loss_fun(y_pred, y_true)
    
    # backward pass
    loss.backward()
    
    # update parameters
    optimizer.step()
    
    # optional: during training keep track of progress
    for name, param in model.named_parameters():
        if param.requires_grad:
            if name == 'linear.weight':
                slope.append(param.data.numpy()[0][0])
            if name == 'linear.bias':
                bias.append(param.data.numpy()[0])
    
    # store losses
    losses.append(float(loss.data))
    
    print(f"Epoch: {epoch}")
    
# %%
sns.lineplot(x=range(NUM_EPOCHS), y=losses)
# %%
sns.lineplot(x=range(NUM_EPOCHS), y=slope)
# %%
sns.lineplot(x=range(NUM_EPOCHS), y=bias)
# %%
