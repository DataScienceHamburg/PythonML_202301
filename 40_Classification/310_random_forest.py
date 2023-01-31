#%% packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
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
    ('random_forest', RandomForestClassifier(n_estimators = 1000, random_state = 42, bootstrap=True))
]

pipeline = Pipeline(steps)

# train the Decision Tree
clf = pipeline.fit(X_train, y_train)

# prediction for Test data
y_pred = clf.predict(X_test)

#%%
y_pred_class = [round(i, 0) for i in y_pred]
cm = confusion_matrix(y_test, y_pred_class)
cm
# %%
accuracy_score(y_true=y_test, y_pred=y_pred_class) * 100
# %% Variable Importance
importances = list(RandomForestClassifier(n_estimators = 1000, random_state = 42, bootstrap=True).fit(X_train, y_train).feature_importances_)

# %%
feature_names = X_train.columns
feature_names

# create a list with feature importances
feature_importance = pd.DataFrame([(feature_names, round(importances, 2)) for feature_names, importances in zip(feature_names, importances)],
                                  columns=['feature', 'importance'])

# sort the importances by most importance first
feature_importance = feature_importance.sort_values(by=['importance'], ascending=False)

g = sns.barplot(data=feature_importance, x='feature', y='importance')
g.set_xticklabels(labels=feature_importance['feature'], rotation=90)  # set x tick-labels vertically
feature_importance
plt.show()

# %%
