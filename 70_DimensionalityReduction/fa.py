#%% package import
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import FactorAnalysis
from factor_analyzer.factor_analyzer import calculate_kmo

from plotnine import ggplot, aes, geom_point, labs
import matplotlib.pyplot as plt
# %% data prep
iris = datasets.load_iris()

# %% separate independent / dependent features
X, y = iris.data, iris.target

# %% Test ob brauchbar für FA
kmo_variable, kmo_total = calculate_kmo(X)
print(f"KMO Total: {kmo_total}")
# %%
steps = [
    ('scaler', StandardScaler()),
    ('fa', FactorAnalysis(n_components=2, random_state=123))
]
pipeline = Pipeline(steps)
factors = pipeline.fit_transform(X)

# %%
df_factors = pd.DataFrame(factors, columns=['F1', 'F2'])
df_factors['target'] = [str(i) for i in y]
# %%
ggplot(data=df_factors) + aes(x='F1', y='F2', color='target') + geom_point()
# %%
