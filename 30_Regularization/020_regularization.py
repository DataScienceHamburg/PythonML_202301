# %% packages
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
# %% Data Import
col = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
housing = pd.read_csv('../data/housing.csv', delim_whitespace=True, names=col)


print("Boston housing dataset has {} data points with {} variables each.".format(*housing.shape))
# %% Data Prep
# Boston Housing dataset is used for predictions of housing prices. It has these independent features:

# - RM...average number of rooms per dwelling
# - LSTAT...% lower status of the population
# - PTRATIO...pupil-teacher ratio by town

# The dependent feature is:

# - MEDV...Median value of owner-occupied homes in $1000's

#%% separate independent / dependent features
X = housing.drop('MEDV', axis=1)
y = housing['MEDV']
# %% train / test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123, test_size=0.4)
# %% Modeling
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

print('Training score: {}'.format(lr_model.score(X_train, y_train)))
print('Test score: {}'.format(lr_model.score(X_test, y_test)))
# %% 2nd order modeling to improve results
steps = [
    ('scalar', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2)),
    ('model', LinearRegression())
]

pipeline = Pipeline(steps)

pipeline.fit(X_train, y_train)

print('Training score: {}'.format(pipeline.score(X_train, y_train)))
print('Test score: {}'.format(pipeline.score(X_test, y_test)))

# This already improved the performance a lot.

# %% Ridge Regression (L2 Regularization)
steps = [
    ('scalar', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2)),
    ('model', Ridge(alpha=10, fit_intercept=True))
]

ridge_pipe = Pipeline(steps)
ridge_pipe.fit(X_train, y_train)

print('Training Score: {}'.format(ridge_pipe.score(X_train, y_train)))
print('Test Score: {}'.format(ridge_pipe.score(X_test, y_test)))

# %% Lasso Regression (L1 Regularization)
steps = [
    ('scalar', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2)),
    ('model', Lasso(alpha=0.3, fit_intercept=True))
]

lasso_pipe = Pipeline(steps)

lasso_pipe.fit(X_train, y_train)

print('Training score: {}'.format(lasso_pipe.score(X_train, y_train)))
print('Test score: {}'.format(lasso_pipe.score(X_test, y_test)))
 
# %%
# This result is better with Ridge-Regression.