#%% package 
import pandas as pd
import numpy as np
from plotnine import ggplot, aes, geom_point, geom_smooth, geom_text
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
#%% data import
starwars = pd.read_csv('../data/Starwars.csv')
# %%
starwars.head()
# %%
starwars.shape
# %%
def plot_height_mass(df):
    g = ggplot(df) + aes(x='height', y='mass') + geom_point() + geom_smooth(method='lm')
    return g
# %%
plot_height_mass(starwars)
# %%
starwars[starwars['mass'] >= 1000]
# %%
starwars_filt = starwars[starwars['mass'] <= 1000]
# %%
plot_height_mass(starwars_filt)
# %% Modeling
X = np.array(starwars_filt['height']).reshape(-1, 1)
y = np.array(starwars_filt['mass']).reshape(-1, 1)
# %% train the model
lin_reg = LinearRegression()
lin_reg.fit(X, y)
# %% create predictions
y_pred = lin_reg.predict(X).reshape(-1)
starwars_filt['y_pred'] = y_pred
# %%
ggplot(starwars_filt) + aes(x='height', y='mass', label='name') + geom_point() + geom_smooth(method='lm') + geom_point(aes(y= 'y_pred'), color='red') + geom_text(size=6)
# %%
R2 = r2_score(y_true=y, y_pred=y_pred)
print(f"R2: {R2}")
# %%
lin_reg.coef_
# %%
lin_reg.intercept_
# %%
