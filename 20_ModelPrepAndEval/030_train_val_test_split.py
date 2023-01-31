#%% packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# %%
df = pd.DataFrame(np.arange(1500).reshape((500, 3)), columns=['x1', 'x2', 'y'])
df.head()
# %% Train / Test Split
X, y = df.loc[:, df.columns != 'y'], df['y']  # separate independent (X) and dependent (y) features

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# %%
print(f"X_train shape: {X_train.shape}\nX_test shape:  {X_test.shape}\ny_train shape: {y_train.shape}\ny_test shape: {y_test.shape}")
# %%
