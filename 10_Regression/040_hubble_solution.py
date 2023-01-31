# %% Hubble Exercise
# In this tutorial you will take a look at measurements of Hubble (the telescope). Besides taking beautiful pictures, it measured speed and distance of Super-Novae. Similar data was used in 1929 by Hubble (the person) and he found out that there is a linear relationship.


# He discovered that galaxies appear to move away. This can be visualised with red-shift of spectral lines. This observation was the first indication that the universe expands.

# You will create a linear model based on observations and create predictions.

# Steps:
#%% 0. Import required packages
import numpy as np
import pandas as pd
from plotnine import ggplot, aes, geom_point, geom_smooth
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

#%% 1. Import the data
hubble = pd.read_csv('../data/Hubble.csv')
hubble.head()
#%% 2. visualise the data for better understanding
(ggplot(hubble) 
 + aes(x='v', y='D') 
 + geom_point()
 + geom_smooth(method='lm', color ='blue', se = False)
)

#%% 3. Create a linear regression model
X_train = np.array(hubble['v']).reshape(-1, 1)
y_train = np.array(hubble['D']).reshape(-1, 1)

regressor = LinearRegression()
regressor.fit(X_train, y_train) 
#%% 4. create predictions
hubble['y_pred'] = regressor.predict(X_train)

#%% 5. calculate the $R^2$ metric
coefficient_of_dermination = r2_score(y_train, hubble['y_pred'])
coefficient_of_dermination

#%% 6. Bonus: calculate the Hubble Constant: $H_0=\frac{v}{D}$
hubble['H0'] = hubble['v'] / hubble['D']
np.mean(hubble['H0'])
# %% 
# You can compare this to most recent observed values, as shown in [this](https://en.wikipedia.org/wiki/Hubble%27s_law) Wikipedia article