#%% packages
# data prep
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

# modeling
from sklearn.cluster import KMeans

# visualisation
from plotnine import ggplot, aes, geom_point, labs
import matplotlib.pyplot as plt
# %%
X, y = make_blobs(n_samples=1000, centers=3, cluster_std=1, random_state=125)
# %%
df = pd.DataFrame(X, columns=['x1', 'x2'])
df['y'] = y
# %% visualisation
ggplot(df) + aes(x='x1', y='x2', color='y') + geom_point()

# %% Modeling
kmeans = KMeans(n_clusters=3, random_state=123)
kmeans.fit(X)

# %%
centers = pd.DataFrame(kmeans.cluster_centers_, columns=['x1', 'x2'])
df['y_kmeans'] = kmeans.predict(X)
# %%
ggplot(df) + aes(x='x1', y='x2', color='y_kmeans') + geom_point() + geom_point(data=centers, color='red', size = 2)
# %% Find optimum cluster number (Elbow method)
distortions = []
for k in range(1, 6):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(df[['x1', 'x2']])
    distortions.append(kmeans.inertia_)
    
# %% 
import seaborn as sns
x_values = list(range(1, len(distortions)+1))
sns.lineplot(x=x_values, y=distortions)
plt.show()
# %%
