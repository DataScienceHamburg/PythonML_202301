#%%
import torch
# %% set required grad bei Erstellung
x = torch.tensor(5.5, requires_grad=True)
x.requires_grad
# %% Setze mittels requires_grad_()
x = torch.tensor(5.5)
x.requires_grad_()
x.requires_grad
# %%
x = torch.tensor(2.0, requires_grad=True)
y = (x-3) * (x-6) * (x-4)

# %%
y.backward()
# %%
x.grad
# %%
import numpy as np
import seaborn as sns
def y_function(val):
    return (val-3) * (val-6) * (val-4)

x_range = np.linspace(0, 10, 101)
x_range
y_range = [y_function(i) for i in x_range]
sns.lineplot(x = x_range, y = y_range)
# %%
