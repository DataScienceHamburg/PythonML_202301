#%% packages
import pandas as pd 
import numpy as np 
from sklearn.metrics import classification_report 
from sklearn.model_selection import train_test_split 
from sklearn.datasets import make_blobs
from sklearn.ensemble import GradientBoostingClassifier

import seaborn as sns
import matplotlib.pyplot as plt
# %% Data Prep
X, y = make_blobs(random_state=0)
# %%
df = pd.DataFrame(X)
df.columns = ['x1', 'x2']
df['y_true'] = y

sns.scatterplot(data=df, x='x1', y='x2', hue='y_true')
plt.show()
# %% Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.33, random_state=42)
# %% Modeling
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,  max_depth=1, random_state=0).fit(X_train, y_train)
# %% Model Evaluation
clf.score(X_test, y_test)
# %%
