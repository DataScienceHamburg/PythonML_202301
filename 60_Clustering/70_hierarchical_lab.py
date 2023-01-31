# %% Data Preparation
import numpy as np
import pandas as pd
import random

# Modeling
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

# Visualisation
from matplotlib import pyplot as plt
from plotnine import ggplot, aes, geom_point, geom_text

# %% Data Preparation
num_points = 10
x = random.sample(population=set(np.linspace(start=0, stop=10, num=num_points)), k=num_points)
y = random.sample(population=set(np.linspace(start=0, stop=10, num=num_points)), k=num_points)
labels = range(1, num_points+1)
df = pd.DataFrame(list(zip(x, y, labels)), columns=['x', 'y', 'point_labels'])


# %%
(ggplot(data=df)
  + aes(x='x', y='y', label='point_labels')
  + geom_point(size=0)
  + geom_text(size=20)
)

# %% Modeling
X = np.array(df[['x', 'y']])
X

linked = linkage(X, 'single')

plt.figure(figsize=(10, 7))
dendrogram(linked,
            orientation='top',
            labels=labels,
            distance_sort='descending',
            show_leaf_counts=True)
plt.show()


# %% Modeling
cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
cluster.fit_predict(X)


# %%
df['class_labels'] = cluster.labels_  # assign the result to a new column
df['class_labels'] = df['class_labels'].astype('category')
(ggplot(data=df)
  + aes(x='x', y='y', color='class_labels', label='point_labels') 
  + geom_point(size = 0)
  + geom_text(size=20)
)





# %%
