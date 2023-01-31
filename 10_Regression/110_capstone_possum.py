#%%
#%% package import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

#%% data understanding
# source: https://www.kaggle.com/code/vrinja/possum-linear-regression/data

# %% data import
possum = pd.read_csv('../data/possum.csv')
# %%
possum_corr = possum.corr()
plt.figure(figsize=(10,10))
sns.heatmap(possum_corr, annot=True)
plt.show()
#%% 
possum.shape
# %%
possum.info()
# %% Missing Data
# find the missing data
print(possum.isnull().sum())
possum_filt = possum.dropna()
use_cols = ['age', 'hdlngth', 'skullw', 'totlngth',
            'taill', 'footlgth', 'earconch', 'eye', 
            'chest', 'belly']
possum_filt = possum_filt[use_cols]
print(possum_filt.shape)

# %% separate dependent and independent features
X = possum_filt.drop(columns=['totlngth'])
y = possum_filt['totlngth']

# %% Train / Test Split
X_train, X_test , y_train , y_test = train_test_split(X, y , test_size =0.15 , random_state=123)

# %% scale the data
scaler = StandardScaler()
# scale train
X_train_scaled = scaler.fit_transform(X_train)
# apply scaling from train to test
X_test_scaled = scaler.transform(X_test)
# %% Modeling
lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_train)

# %% create preds
y_pred_test = lin_reg.predict(X_test_scaled)

# %%
sns.regplot(x=y_test, y=y_pred_test)
# %%
print(f"R**2 score: {r2_score(y_test, y_pred_test)}")
# %% get the model coefficients
lin_reg.coef_

# %%
