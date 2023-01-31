#%% Data Understanding

# Wine variants of Portuguese "Vinho Verde" are analysed with regards to their chemical properties. Finally, we are interested how these chemical properties influence wine quality.


# These are our independent variables:

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

# This is our dependent variable:

# 12. quality (score between 0 and 10)

#%% Packages
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import seaborn as sns
import matplotlib.pyplot as plt
# %%
wine = pd.read_csv('../data/winequality-red.csv', sep=';')
wine.head()
# %% 
wine.describe()

# %% visualise the data
sns.pairplot(wine.iloc[:, 7:12], hue='quality')
plt.show()
# %% calculate correlation matrix
wine.corr()

#%% Modeling
X_train = np.array(wine.loc[:, wine.columns != 'quality'])
y_train = np.array(wine['quality']).reshape(-1, 1)
# %%
regressor = LinearRegression()
regressor.fit(X_train, y_train) 
# %% create predictions
wine['quality_pred'] = regressor.predict(X_train)

#%%
sns.scatterplot(data=wine, x='quality', y='quality_pred')
plt.show()
# %% calculate metric
r2_score(wine['quality'], wine['quality_pred'])

# %%
