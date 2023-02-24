#%% packages
import numpy as np
import pandas as pd
import torch
import torch.nn as nn 
import seaborn as sns
from torch.utils.data import Dataset, DataLoader

#%% data import
cars_file = 'https://gist.githubusercontent.com/noamross/e5d3e859aa0c794be10b/raw/b999fb4425b54c63cab088c0ce2c0d6ce961a563/cars.csv'
cars = pd.read_csv(cars_file)
cars.head()

#%% visualise the model
sns.scatterplot(x='wt', y='mpg', data=cars)
sns.regplot(x='wt', y='mpg', data=cars)

#%% Hyperparameter
BATCH_SIZE = 2

#%% convert data to tensor
X_list = cars.wt.values
X_np = np.array(X_list, dtype=np.float32).reshape(-1,1)
y_list = cars.mpg.values
y_np = np.array(y_list, dtype=np.float32).reshape(-1,1)
X = torch.from_numpy(X_np)
y_true = torch.from_numpy(y_np)

#%% Dataset / Dataloader
class LinearRegressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
lin_reg_ds = LinearRegressionDataset(X_np, y_np)
train_loader = DataLoader(dataset=lin_reg_ds, batch_size=BATCH_SIZE)

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
NUM_EPOCHS = 2000

losses, slope, bias = [], [], []

for epoch in range(NUM_EPOCHS):
    for j, (X, y) in enumerate(train_loader):
        
        # set grads to zero
        optimizer.zero_grad()
        
        # forward pass
        y_pred = model(X)
        
        # calc loss
        loss = loss_fun(y_pred, y)
        
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
# %% Save Model
torch.save(model.state_dict(), 'models/model_v1.pt')

# %% Load Model
model_reloaded = LinearRegressionTorch(input_size=input_dim, output_size=output_dim)
model_reloaded.load_state_dict(torch.load('models/model_v1.pt'))

# %%
