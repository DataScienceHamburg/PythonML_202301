#%% packages
# data handling
import numpy as np
import pandas as pd

# modeling
from sklearn.cluster import KMeans

# prepare sample data
from sklearn.datasets import make_blobs

# data visualisation
from plotnine import ggplot, aes, geom_point, labs
# %% Data Prep
X, y_true = make_blobs(n_samples=1000, 
                       centers=3,
                       cluster_std=1, 
                       random_state=123)

df = pd.DataFrame(X)
df.columns = ['x', 'y']
df['y_true'] = [str(i) for i in y_true.tolist()]

(ggplot(df) + 
    aes(x='x', y='y', color=y_true) +
    geom_point() +
    labs(x='x', y='y', color='target')
)

#%% Modeling
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

# get cluster centers
centers = pd.DataFrame(kmeans.cluster_centers_)
centers.columns = ['x', 'y']

# create predictions
df['y_kmeans'] = kmeans.predict(X)
df['y_kmeans'] = df['y_kmeans'].astype("category")

# visualise results
(ggplot(data=df, mapping=aes(x='x', y='y')) +
    geom_point(mapping=aes(color='y_kmeans')) +
    geom_point(data=centers, mapping=aes(x='x', y='y'), color='red', size=2) +
    labs(x='x', y='y', color='target')
) 
# %%
