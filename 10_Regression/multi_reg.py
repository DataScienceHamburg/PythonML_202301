#%% packages
#%% package import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import seaborn as sns


# %%

# 1. fixed acidity 
# 2. volatile acidity
# 3. citric acid 
# 4. residual sugar 
# 5. chlorides 
# 6. free sulfur dioxide 
# 7. total sulfur dioxide 
# 8. density 
# 9. pH 
# 10. sulphates 
# 11. alcohol 

# 12. quality (score between 0 and 10)

wine = pd.read_csv('../data/winequality-red.csv', sep=";")
# %%
wine
# %%
wine.describe()
# %% visualise the data
sns.pairplot(wine.iloc[:, 7:12], hue='quality')
plt.show()

# %%
corr = wine.corr()
# %%
sns.heatmap(corr, vmin=-1, vmax=1, annot=np.round(corr, 2))
plt.show()
# %% Modeling
X = np.array(wine.loc[:, wine.columns != 'quality'])
y = np.array(wine['quality']).reshape(-1, 1)
print(f"X shape: {X.shape}, y shape: {y.shape}")
# %%
lin_reg = LinearRegression()
lin_reg.fit(X, y)
# %%
lin_reg.coef_
# %%
lin_reg.intercept_
# %%
wine['y_pred'] = lin_reg.predict(X)
# %% correlation plot
sns.scatterplot(data=wine, x='quality', y='y_pred')
plt.show()
# %%
r2_score(y_pred=wine['y_pred'], y_true=wine['quality'])
# %%
