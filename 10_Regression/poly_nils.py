#%% package import
import numpy as np
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
from plotnine import ggplot, aes, geom_point, geom_smooth
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
 
#%%
sample_data = pd.DataFrame({"x":np.arange(-20,40,0.5)})
sample_data["y"] = 50+0.25*(sample_data["x"]-5)**3
sample_data["y_noise"] = sample_data["y"]+np.random.normal(loc=100,scale=500, size=sample_data.shape[0])
sample_data
 
#%%
sample_data.plot(x="x", y="y_noise", kind="scatter")
 
##% 1. Separate Data into independent and dependent
x = np.array(sample_data["x"]).reshape(-1,1)
y = np.array(sample_data["y_noise"]).reshape(-1,1)
 
# %% 2. Model fitting
degree = 3
poly_feat = PolynomialFeatures(degree=degree)
x_poly = poly_feat.fit_transform(x)
 
model = LinearRegression()
model.fit(x_poly,y)
 
# %%
sample_data["y_pred"] = model.predict(x_poly)
y_pred = model.predict(x_poly)
 
#%%
sample_data.plot(x="x", style="o")
 
#%%
r2 = r2_score(y_true=y, y_pred=y_pred.reshape(-1))
r2
# %%
