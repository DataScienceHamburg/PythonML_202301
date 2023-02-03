#%% package
#%% package import
#%% package import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plotnine import ggplot, aes, geom_point, geom_smooth
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
# %% prepare sample data
sample_data = pd.DataFrame({'x': np.arange(-20, 40, 0.5)})
sample_data['y'] = 50 + 0.25 * (sample_data['x'] - 5)**3
sample_data['y_noise'] = sample_data['y'] + np.random.normal(loc=100, scale=500, size =sample_data.shape[0])
# %%
ggplot(sample_data) + aes(x='x', y='y_noise') + geom_point()
# %% Modeling

# 1. Separate data into independent and dependent
X = np.array(sample_data['x']).reshape(-1, 1)
y = np.array(sample_data['y_noise']).reshape(-1, 1)

# 2. Model Fitting
degree = 3
poly_feat = PolynomialFeatures(degree=degree)
x_poly = poly_feat.fit_transform(X)

model = LinearRegression()
model.fit(x_poly, y)


# %%
y_pred = model.predict(x_poly)
sample_data['y_poly_pred'] = y_pred
# %%
ggplot(sample_data) + aes(x='x', y='y_noise') + geom_point() + geom_point(aes(y='y_poly_pred'), color='red')
# %%
r2_score(y_pred, y)
# %% reshape
data = np.array([2, 2, 2, 2, 2, 2, 2, 2])

# %%
data.shape
# %%
data.reshape(-1, 2, 2)
# %%
