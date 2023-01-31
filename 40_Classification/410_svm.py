#%% packages
import numpy as np
import pandas as pd
import random

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC  # SVMs for Classification

# Visualisation
import seaborn as sns
import matplotlib.pyplot as plt
# %% Data Prep
x = random.sample(population=set(np.linspace(start=-10, stop=10, num=100000)), k=1000)
y = random.sample(population=set(np.linspace(start=-10, stop=10, num=100000)), k=1000)
z = [(x**2 + y**2) for x, y in zip(x, y)]

df = pd.DataFrame(list(zip(x, y, z)), columns=['x', 'y', 'z'])
df['class'] = [1 if i<50 else 0 for i in df['z']]
# %%
sns.scatterplot(data=df, x='x', y='y', hue='class')
plt.show()
# %% Separate Independent / Dependent Features
X = df[['x', 'y', 'z']]
y = df['class']

#%% Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)
# %% Modeling
clf = SVC(kernel='rbf')
clf.fit(X_train, y_train)

# %% create preds
y_test_pred = clf.predict(X_test)

df_test = pd.DataFrame(X_test)
df_test['y_test_pred'] = y_test_pred
df_test['y_test'] = y_test
# %%
sns.scatterplot(data=df_test, x='x', y='y', hue='y_test_pred')
plt.show()
# %%
confusion_matrix(y_test, y_test_pred)

# %%
accuracy_score(y_true=y_test, y_pred=y_test_pred)
# %%
