#%%
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.manifold import TSNE
from plotnine import ggplot, aes, geom_point
# %% Mnist
mnist = datasets.load_digits()

# %%
X, y = mnist.data, mnist.target
# %%
digits_embedded = TSNE(n_components=2).fit_transform(X)
# %%
df = pd.DataFrame(digits_embedded, columns=['Component1', 'Component2'])
df['target'] = pd.Categorical(y)
# %%
ggplot(df) + aes(x='Component1', y='Component2', color='target') + geom_point()
# %%
