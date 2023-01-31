#%% packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
#%% Data Understanding
# This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective of the dataset is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset. Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage.

# %% Data Import
diabetes = pd.read_csv('../data/diabetes.csv')

# %% Exploratory Data Analysis
diabetes.head()

# %%
diabetes.describe()
# %% correlation matrix
corr = diabetes.corr()
# generate mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0, linewidths=.5)
plt.show()
# %% Train / Test Split
X = diabetes.drop(['Outcome'], axis=1)
y = diabetes['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
# %% Modeling
steps = [
    ('scaler', StandardScaler()),
    ('decision_tree', DecisionTreeClassifier())
]

pipeline = Pipeline(steps)

# train the Decision Tree
clf = pipeline.fit(X_train, y_train)

# prediction for Test data
y_pred = clf.predict(X_test)

# %% Model Evaluation
cm = confusion_matrix(y_test, y_pred)
cm

# %%
accuracy_score(y_true=y_test, y_pred=y_pred)
# %% Visualise Decision Tree
from sklearn import tree
tree.plot_tree(clf)
# %%
plt.show()
