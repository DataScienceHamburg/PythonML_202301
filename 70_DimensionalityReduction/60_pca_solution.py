# Intro: You will use the "digits" dataset and apply PCA to it


#%% Packages
# 1. Import required packages
# Data Preparation
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits

# Modeling
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# Visualisation
from plotnine import *


# # Data Preparation


#%% 2. Load required data
digits = load_digits()
digits_data = digits.data
digits_data.shape


# # Modeling

#%% 3. Apply PCA and reduce the number of dimensions to two!
pca = PCA(2)  # project from 64 to 2 dimensions
projected = pca.fit_transform(digits_data)
print("Shape of Digits: %s" % str(digits_data.shape))
print("Shape of Projection: %s" % str(projected.shape))

df_proj = pd.DataFrame(projected, columns=['PC1', 'PC2'])
df_proj['true_label'] = pd.Categorical(digits.target)


#%% 4. Visualise the result with a graphic framework of your choice.

(ggplot(data=df_proj)
 + aes(x='PC1', y='PC2', color='true_label')
 + geom_point()
)


