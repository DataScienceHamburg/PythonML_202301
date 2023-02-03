#%% package
#%% package import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut, cross_val_score
from sklearn.metrics import r2_score
import seaborn as sns
# %%
wine = pd.read_csv('../data/winequality-red.csv', sep=';')
wine.shape
# %% Separate / Independent features
target_feature = 'quality'
X, y = np.array(wine.loc[:, wine.columns != target_feature]), np.array(wine.loc[:, wine.columns == target_feature])

# %%
print(f"{X.shape}, {y.shape}")
# %% Modeling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_test_pred = lin_reg.predict(X_test)
r2_score(y_true=y_test, y_pred=y_test_pred)

# %% 10-fold cross validation
kf = KFold(n_splits=10, shuffle=True)
kf.get_n_splits(X)
print(kf)

# %%
scores = []
model = LinearRegression()
for train_index, test_index in kf.split(X):
    print(f"{train_index.shape}, {test_index.shape}")
    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    model.fit(X_train, y_train)
    scores.append(model.score(X_test, y_test))
# %%
scores
# %%
np.median(scores)
# %% LOOCV
loocv = LeaveOneOut()
loocv.get_n_splits(X)

y_test_preds, y_test_trues = [], []
scores = []
model = LinearRegression()
for train_index, test_index in loocv.split(X):
    print(f"{train_index.shape}, {test_index.shape}")
    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)[0][0]
    y_test_true = y_test[0][0]
    y_test_preds.append(y_test_pred)
    y_test_trues.append(y_test_true)

# %%
r2_score(y_true = y_test_trues, y_pred = y_test_preds)
# %%
