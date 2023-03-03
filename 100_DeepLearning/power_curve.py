#%% packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader 
from plotnine import ggplot, aes, geom_point
import seaborn as sns
import matplotlib.pyplot as plt
from skorch import NeuralNetRegressor
from sklearn.model_selection import GridSearchCV
# %%
file_path = 'data/Turbine_Data.csv'
df = pd.read_csv(file_path, sep=",")

#%% clean dataset
df_filt = df.dropna().copy()
df_filt.shape
# %% visualise data
ggplot(df_filt) + aes(x='WindSpeed', y='ActivePower') + geom_point(alpha = 0.01)
# %% independent and dependent features
X = df_filt[['WindSpeed', 'AmbientTemperatue', 'RotorRPM']]
y = df_filt['ActivePower']

# %% train test split
X_train, X_test, y_batch, y_test = train_test_split(X, y, test_size = 0.2)

#%% Dataset und Dataloader
class PcData(Dataset):
    def __init__(self, X, y) -> None:
        super().__init__()
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32)
        self.len = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
    def __len__(self):
        return self.len
    
data_train = PcData(X_train, y_batch)
data_test = PcData(X_test, y_test)

train_loader = DataLoader(dataset=data_train, batch_size=32)
test_loader = DataLoader(dataset=data_test, batch_size=32)
# %%
print(f"X_train Shape: {data_train.X.shape}, y_train shape: {data_train.y.shape}")
print(f"X_test Shape: {data_test.X.shape}, y_test shape: {data_test.y.shape}")

# %% Model Class
class PowerCurveNet(nn.Module):
    def __init__(self, NUM_INPUT_FEATURES, NUM_HIDDEN_FEATURES):
        super().__init__()
        self.lin1 = nn.Linear(NUM_INPUT_FEATURES, NUM_HIDDEN_FEATURES)
        self.lin2 = nn.Linear(NUM_HIDDEN_FEATURES, 1)
        
    def forward(self, x):
        x = self.lin1(x)
        x = torch.relu(x)
        x = self.lin2(x)
        return x
    

model = PowerCurveNet(NUM_INPUT_FEATURES=X_train.shape[1], NUM_HIDDEN_FEATURES=100)

# %%
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())
# %% Training Loop
NUM_EPOCHS = 20
losses = []

for e in range(NUM_EPOCHS):
    for X_batch, y_batch in train_loader:
        # init gradients
        optimizer.zero_grad()

        # forward pass
        y_pred = model(X_batch)
        
        # calc loss
        loss = loss_fn(y_pred[:, 0], y_batch)
        
        # calc grads
        loss.backward()
        
        # update pars
        optimizer.step()
    
    # add losses
    losses.append(float(loss.data.detach().numpy()))
    
    print(f"Epoch {e}")
#%% Loss over Epoch
sns.lineplot(x= range(len(losses)), y = losses)  
plt.show()
    
# %% calc predictions on test data
X_tests_ws = []
y_tests = []
y_test_preds = []
for X_test_batch, y_test_batch in test_loader:
    with torch.no_grad():
        y_test_pred = model(X_test_batch)
        X_tests_ws.extend(X_test_batch[:, 0].numpy())
        y_test_preds.extend(y_test_pred.numpy()[:, 0])
        y_tests.extend(y_test_batch.numpy())
        
        
# %%
sns.scatterplot(x=y_tests, y=y_test_preds)
plt.show()
# %%
sns.scatterplot(x=X_tests_ws, y=y_test_preds, s=10)
plt.show()
# %%
sns.scatterplot(x=X_tests_ws, y=y_tests, s=10)
plt.show()
# %%
net = NeuralNetRegressor(PowerCurveNet(NUM_INPUT_FEATURES=X_train.shape[1], NUM_HIDDEN_FEATURES=100), max_epochs=10, lr = 0.01)
net.set_params(train_split=False, verbose=0)

params = {
    #'lr': [0.001],
    'max_epochs': [10, 50]
}

gs = GridSearchCV(net, params, refit=False, cv=2, scoring='r2', verbose=2)

X_torch = torch.tensor(np.array(X), dtype=torch.float32)
y_torch = torch.tensor(np.array(y).reshape(-1, 1), dtype=torch.float32)
gs.fit(X_torch, y_torch)

#%%
gs.best_score_

# %%
gs.best_params_
# %%
