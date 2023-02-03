#%% package
#%% package import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# %%
data = np.arange(1500).reshape(-1, 3)
data.shape
df = pd.DataFrame(data, columns=['x1', 'x2', 'y'])
df
# %% separate indepependent / dependent feature
# X = df.iloc[:, :2]
X, y = df.loc[:, df.columns != 'y'], df['y']
print(f"Shapes X: {X.shape}, y: {y.shape}")
# %% train / test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=123)
y_train

# %%
