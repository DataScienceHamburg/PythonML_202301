# %% Factor Analysis
#%% Packages
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import FactorAnalysis
from plotnine import ggplot, aes, geom_point, scale_color_discrete, labs
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt

#%% Data Preparation
# We will work with the **iris** dataset. It is shipped with *sklearn*.

iris = datasets.load_iris()

X = iris.data
y = iris.target


# %%
iris.target_names


# %%
X = pd.DataFrame(X, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])

print(X.head())  # head of dataframe
print(X.columns)  # columns
print(X.shape)  # object shape

# %% Factor Analysis

#%% Test for Factorability 
print('Bartlett-sphericity Chi-square: {}'.format(calculate_bartlett_sphericity(X)[0]))
print('Bartlett-sphericity P-value: {}'.format(calculate_bartlett_sphericity(X)[1]))

kmo_all, kmo_model = calculate_kmo(X);
print('KMO score: {}'.format(kmo_model));

# for good KMO value should be around 0.6

#%% Number of Factors
fa = FactorAnalyzer()
fa.fit(X, 10)
ev, v = fa.get_eigenvalues()
plt.plot(range(1,X.shape[1]+1),ev)
plt.show()
# %% Modeling
steps = [
    ('scalar', StandardScaler()),
    ('fa', FactorAnalysis(n_components=2, random_state=123))
]

pipeline = Pipeline(steps)

factors = pipeline.fit_transform(X)

# %% [markdown]
# We want to have a target vector with names instead of numbers. To achieve this, we need to create a mapping, and then use list comprehension to create a new list *y_strings*.

# %%
mapping = {0: 'setosa', 1:'versicolor', 2: 'virginica'}
y_strings = y.astype(int)
y_strings = [mapping[y[i]] for i in range(len(y_strings))]


# %%
factors_df = pd.DataFrame(factors, columns=['F1', 'F2'])
factors_df['target'] = y_strings


# %%
(ggplot(data=factors_df) 
    + aes(x='F1', y='F2', color='target') 
    + geom_point()
    + labs(x='Factor 1', y='Factor 2', title='Factor Analysis for Iris Dataset')
    + scale_color_discrete(name='Iris Class')
)


# %%
factors




