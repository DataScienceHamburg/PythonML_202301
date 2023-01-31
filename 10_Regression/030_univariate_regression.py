#%% packages
import pandas as pd
import numpy as np
from plotnine import ggplot, aes, geom_point, geom_smooth, geom_text
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import r2_score
# %% Introduction
# Based on star wars characters we will predict person weight based on its height.

# What is independent and dependent variable? Well, the height is more or less defined in our genes, but our weight can be influenced. So I use height as independent variable ad mass as dependent variable.

#%% Data Import
starwars = pd.read_csv('../data/Starwars.csv')
starwars.head()
# %% Shape
starwars.shape

# %% visualise the height and mass
def plot_height_mass(df):
    g = (ggplot(df)
 + aes(x='height', y = 'mass') 
 + geom_point()
 + geom_smooth(method = 'lm')
)
    return g

plot_height_mass(starwars)


# %% outlier
starwars[starwars['mass']>=1000]

# %% filter outlier
starwars_filt = starwars[starwars['mass']<1000]
plot_height_mass(starwars_filt)
# %%
starwars_filt.shape
# %% Modeling
X_train = np.array(starwars_filt['height']).reshape(-1, 1)
y_train = np.array(starwars_filt['mass']).reshape(-1, 1)

# %%
regressor = LinearRegression()
regressor.fit(X_train, y_train) 
# %% Create Predictions
y_pred = regressor.predict(X_train).reshape(-1)
starwars_filt['y_pred'] = y_pred
# %% visualise result
(ggplot(starwars_filt)
 + aes(x='height', y = 'mass', label = 'name') 
 + geom_point()
 + geom_text()
 + geom_smooth(method = 'lm')
 + geom_point(aes(y='y_pred'), color = 'red')
)
# %% calculate metrics
coefficient_of_dermination = r2_score(y_train, y_pred)
coefficient_of_dermination

# %%
