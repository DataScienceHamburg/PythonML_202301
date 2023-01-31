# %% Required Packages
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from plotnine import ggplot, aes, geom_point, labs, theme_bw
import seaborn as sns
import matplotlib.pyplot as plt
# %% PCA for Data Visualisation
# PCA can be used to reduce the number of dimensions, so that you can see differences in a lower dimension. If the dimension is two or three you can plot the result.
# %% Data Preparation
# We will work with the **iris** dataset. It is shipped with *sklearn*.

iris = datasets.load_iris()


# %%
iris_data = iris.data


# %%
X = pd.DataFrame(iris_data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
X.head()


# %%
y = iris.target

#%% Modeling
# Standardisation of data, followed by direct implementation of PCA. PCA is performed to reduce the number of dimensions from 4 to 2.
# 
# Scaling of the data is relevant, because features can be of very different ranges.
pca = PCA(n_components=2)

steps = [
    ('scalar', StandardScaler()),
    ('pca', pca)
]

pipeline = Pipeline(steps)

prin_comp = pipeline.fit_transform(X)

#%% PCA Loadings
pca_loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
pca_loading_matrix = pd.DataFrame(pca_loadings, columns =    
                        ['PC{}'.format(i) for i in range(1, 3)], 
                        index=iris.feature_names 
                        )
sns.heatmap(pca_loading_matrix, cmap=sns.diverging_palette(200, 30, n=200), annot=True)
plt.show()
# %%
prin_df = pd.DataFrame(data = prin_comp
             , columns = ['PC1', 'PC2'])

y_df = pd.DataFrame([str(i) for i in y], columns=['y'])

prin_df = pd.concat([prin_df, y_df], axis = 1)


# %%
(ggplot(data=prin_df) 
    + aes(x='PC1', y='PC2') 
    + geom_point(aes(color='y'))
    + labs(title ='Principal Component Analysis for Iris Dataset')
    + theme_bw()
)

# %% [markdown]
# The classes are now much clearer to separate, although we reduced the dimensions from four to two.
# %% [markdown]
# The explained variance can be extracted from the pca object.

# %%
pca.explained_variance_ratio_

# %% [markdown]
# # PCA for Speeding up ML
# %% [markdown]
# Import the data for Mnist. It is part of *sklearn*.

# %%
mnist = datasets.load_digits()


# %%
#images_and_labels = list(zip(mnist.images, mnist.target))


# %%
n_samples = len(mnist.images)
X = mnist.images.reshape((n_samples, -1))
y = mnist.target


# %%
X.shape, y.shape

# %% [markdown]
# Data is splitted into training and testing.

# %%
X_train, X_test, y_train, y_test = train_test_split( mnist.data, mnist.target, test_size=0.2, random_state=0)


# %%
X_train.shape


# %% [markdown]
# **fit()** calculates mean and standard deviation. We need to ensure that these values are derived only from training data, but that the transformations are applied to training and testing.

# %%
scaler = StandardScaler()
pca = PCA(n_components=2)
steps = [
    ('scalar', scaler),
    ('pca', pca)
]

pipeline = Pipeline(steps)

X_train_res = pipeline.fit_transform(X_train)
X_test_res = pipeline.transform(X_test)

# %% [markdown]
# Now we will create a graph that shows PC1 and PC2. The colors indicate the classes (digits). We do this for the training and test data.
# %% [markdown]
# At first the training data

# %%
prin_df = pd.DataFrame(data = X_train_res, columns = ['PC1', 'PC2'])
y_df = pd.DataFrame(y_train, columns=['y'])
y_df['y'] = y_df['y'].astype('category')
prin_df = pd.concat([prin_df, y_df], axis = 1)

(ggplot(data=prin_df) 
    + aes(x='PC1', y='PC2') 
    + geom_point(aes(color='y'))
    + labs(title ='Principal Component Analysis for Mnist Train-Dataset')
    + theme_bw()
)

# %% [markdown]
# Now the test data.

# %%
prin_df = pd.DataFrame(data = X_test_res, columns = ['PC1', 'PC2'])
y_df = pd.DataFrame(y_test, columns=['y'])
y_df['y'] = y_df['y'].astype('category')
prin_df = pd.concat([prin_df, y_df], axis = 1)

(ggplot(data=prin_df) 
    + aes(x='PC1', y='PC2') 
    + geom_point(aes(color='y'))
    + labs(title ='Principal Component Analysis for Mnist Test-Dataset')
    + theme_bw()
)


# %%



