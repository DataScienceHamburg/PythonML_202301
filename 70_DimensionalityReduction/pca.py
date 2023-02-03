#%%
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from plotnine import ggplot, aes, geom_point
import seaborn as sns
# %% data prep
iris = datasets.load_iris()

# %% separate independent / dependent features
X, y = iris.data, iris.target

# %% modeling
pca = PCA(n_components=2)
steps = [
    ('scaler', StandardScaler()),
    ('pca', pca)
]
pipeline = Pipeline(steps)
prin_comp = pipeline.fit_transform(X)
prin_comp
# %% PCA Loadings
pca_loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

# %%
pca_loading_matrix = pd.DataFrame(pca_loadings, columns=['PC1', 'PC2'])
sns.heatmap(pca_loading_matrix, annot=True)
plt.show()

# %%
prin_df = pd.DataFrame(data=prin_comp, columns=['PC1', 'PC2'])
# %%
prin_df['y'] = [str(i) for i in y]
# %%
ggplot(prin_df) + aes(x='PC1', y='PC2', color='y') + geom_point()
# %%
np.sum(pca.explained_variance_ratio_)
# %% Mnist
mnist = datasets.load_digits()

# %%
X, y = mnist.data, mnist.target
# %% train test split
X_train, X_test, y_train, y_test = train_test_split(  X, y, test_size=0.33, random_state=42)


# %%
X_sample = X[200, :].reshape(8,8)
sns.heatmap(X_sample)
plt.show()

# %%
steps = [
    ('scalar', StandardScaler()),
    ('pca', PCA(n_components=2))
]
# %%
pipeline = Pipeline(steps)
X_train_res = pipeline.fit_transform(X_train)
X_test_res = pipeline.transform(X_test)
# %%
prin_df = pd.DataFrame(X_train_res, columns=['PC1', 'PC2'])
prin_df['y'] = [str(i) for i in y_train]
# %%
ggplot(prin_df) + aes(x='PC1', y='PC2', color='y', label ='y') + geom_text()
# %%
